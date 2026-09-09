"""Load A-share data from DuckDB into model-ready tensors.

Price semantics (p19, ``docs/p19_market_price_semantics_contract.md`` §2):
the loader owns the **single** derivation of the dual price basis —

* ``raw_data_cache`` carries the **raw execution prices** (v2
  ``daily_bar_raw``, unadjusted OHLCV + pre_close, nominal yuan), the raw
  ``adj_factor`` column and the exact execution-constraint matrices
  (``up_limit`` / ``down_limit`` from ``daily_price_limit``, ``suspended``
  from ``daily_suspension``);
* ``adjusted_data_cache`` carries the **adjusted factor/label prices**
  (``raw × adj_factor(t) / adj_factor_base``, window-first-day anchor,
  OHLC scaled uniformly) derived by
  :func:`ashare_data.processor.adjusted_price_scale` — the one
  implementation; no ``ffill().bfill()`` continuity is ever fabricated;
* ``target_ret`` and the factor tensor consume the adjusted basis;
  execution (whole lots, cash, commission floor, limit/suspension masks)
  consumes the raw basis and the exact constraint tables (p19 §2.2/§2.4).

The formal price source is the v2 Tushare/PIT database (p17); the legacy
qfq ``daily_bar`` table is a legacy-audit source only and is never read
here.  A missing or schema-incompatible v2 source fails closed.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
import torch
from loguru import logger

from ashare_data.capital_flow import (
    build_capital_frames,
    build_industry_member_frame,
    industry_index_freshness,
)
from ashare_data.gates import FRESHNESS_TOLERANCE_SESSIONS
from ashare_data.config import (
    DataConfig,
    ModelConfig,
    make_data_config,
    resolve_v2_duckdb_path,
)
from ashare_data.db import AshareDB, sql_quoted_list
from ashare_data.fundamentals import build_pit_frames
from ashare_data.manifest import resolve_dataset_id
from ashare_data.processor import (
    adjusted_price_scale,
    apply_adjusted_scale,
    encode_industry_frame,
    is_valid_a_share_code,
    normalize_daily_bars,
    open_to_open_returns,
    pivot_wide,
    exact_tradability_matrix,
)
from ashare_data.tushare_store import V2_TABLES
from ashare_data.universe import (
    ResolvedUniverse,
    UniverseContractError,
    UniverseContractStatus,
    UniverseMask,
    UniversePolicy,
    build_universe_mask,
    dev_fallback_membership_records,
    resolve_universe_contract,
)

from .factors import compute_factor_tensor
from .load_cache import factor_cache_for, loader_factor_cache_identity


class V2PriceSourceError(RuntimeError):
    """The v2 price source is missing or unusable (fail-closed, p19 §2.1)."""


# Cache keys the execution consumers (engine, golden harness, matcher,
# reward path) require — p19 §2.1/§2.4.  Missing keys fail closed.
EXECUTION_CACHE_KEYS = (
    "open",
    "high",
    "low",
    "pre_close",
    "volume",
    "adjusted_open",
    "up_limit",
    "down_limit",
    "suspended",
)

# 0-filled wide columns (missing bar -> 0, the historical cache convention).
_ZERO_FILLED_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "turnover_rate",
    "adj_factor",
)


def _v2_table(table: str) -> str:
    return V2_TABLES[table].name


def apply_cross_sectional_z(
    frames: dict[str, pd.DataFrame], names: tuple[str, ...]
) -> None:
    """Per-date cross-sectional z-score of the named frames, in place.

    F2 (p24 §2): the margin-crowding factors are pre-registered as the
    per-date cross-sectional z of the raw ratios.  The single xcs-z
    implementation (``feature_registry._zscore_per_date``) is reused via
    this frame adapter — no second standardization implementation (§1
    principle 3).  NaN cells (no margin/basic coverage) are restored
    after the z so missing sources stay NaN (never neutral-0; p24 §2 NaN
    guards).
    """

    from .feature_registry import _zscore_per_date

    for name in names:
        frame = frames[name]
        values = frame.to_numpy(dtype=np.float64)
        eligible = np.isfinite(values)
        # The layout (1, stocks, dates): _zscore_per_date computes the
        # cross-sectional z over axis 1 (the stocks) per date — the
        # (stocks, 1, dates) feed would degenerate to z=0 over a
        # size-1 axis.
        z = _zscore_per_date(
            values[None, :, :], eligible[None, :, :]
        )[0, :, :]
        z = np.where(eligible, z, np.nan)
        frames[name] = pd.DataFrame(
            z, index=frame.index, columns=frame.columns
        )


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
        self.dataset_id: str | None = None
        self.factor_tensor: torch.Tensor | None = None
        # p21 §3: how the E2 factor-tensor cache fared on the last load —
        # "miss" (computed + stored), "hit" (restored), "disabled"
        # (identity underivable; pre-wiring path) or None (not loaded).
        self.factor_cache_outcome: str | None = None
        self.industry_codes: torch.Tensor | None = None
        self.raw_data_cache: dict[str, torch.Tensor] = {}
        self.adjusted_data_cache: dict[str, torch.Tensor] = {}
        self.price_degraded: dict[str, str] = {}
        self.industry_index_lag: int | None = None
        self.industry_source_degraded: str | None = None
        self.target_ret: torch.Tensor | None = None
        self.stock_names: dict[str, str] = {}
        self.stock_list_dates: dict[str, object] = {}
        self.current_st_codes: set[str] = set()
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
        """Load names, listing dates and the current ST snapshot.

        ``stocks.is_st`` is a current snapshot: it is exposed as
        :attr:`current_st_codes` and may only be consumed by same-day
        execution (the paper-trading runner); historical paths must never
        use it because it proves no past cell's status.  Failures (e.g.
        the table does not exist yet) leave the maps empty so the loader
        still works on a bare daily-bar database.
        """

        self.stock_names = {}
        self.stock_list_dates = {}
        self.current_st_codes = set()
        try:
            with AshareDB(self.config.duckdb_path, read_only=True) as db:
                df = db.query(
                    f"SELECT ts_code, name, list_date, is_st "
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
            if bool(row.get("is_st") or False):
                self.current_st_codes.add(code)

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
        # membership only; single definition in ashare_data.universe
        # (arch-review F4).  It is never persisted and its degraded
        # provenance remains visible through ``universe_status``.
        return dev_fallback_membership_records(
            self.config.index_codes[0], contract
        )

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

    def _v2_path(self) -> Path:
        """The v2 price source path (single resolution rule, p19 §2.1)."""

        return resolve_v2_duckdb_path(self.config)

    def _load_v2_bars(
        self,
        requested_codes: list[str],
        requested_dates: list[str] | None,
    ) -> pd.DataFrame:
        """Read the v2 price source into one raw long frame (p19 §2.1).

        Single formal price input: ``daily_bar_raw`` (unadjusted OHLCV,
        ``vol`` renamed to the pipeline's ``volume``), ``daily_adj_factor``
        (raw values, never filled), ``daily_price_limit`` (exact
        up/down limits), ``daily_suspension`` (presence -> suspended) and
        ``daily_basic_pit`` (``turnover_rate``).  A missing source file,
        missing table or missing column fails closed.
        """

        path = self._v2_path()
        if not path.exists():
            raise V2PriceSourceError(
                "v2 price source not found: "
                f"{path} (p19 §2.1: the formal loader reads the v2 "
                "Tushare/PIT tables; the legacy qfq daily table is a "
                "legacy-audit source and is never used as a fallback)"
            )
        code_list = sql_quoted_list(requested_codes)
        where_dates = (
            "AND trade_date IN (" + sql_quoted_list(requested_dates) + ")"
            if requested_dates
            else ""
        )

        def _query(db: AshareDB, table: str, columns: str) -> pd.DataFrame:
            try:
                return db.query(
                    f"SELECT {columns} FROM {_v2_table(table)} "
                    f"WHERE ts_code IN ({code_list}) {where_dates} "
                    f"ORDER BY trade_date"
                )
            except duckdb.CatalogException as exc:
                raise V2PriceSourceError(
                    f"v2 price source {path} is missing table "
                    f"{_v2_table(table)!r} (p17 §5 schema required): {exc}"
                ) from exc

        with AshareDB(path, read_only=True) as v2:
            bars = _query(
                v2,
                "daily_bar_raw",
                "ts_code, trade_date, open, high, low, close, pre_close, "
                "vol, amount",
            )
            adj = _query(
                v2, "daily_adj_factor", "ts_code, trade_date, adj_factor"
            )
            limits = _query(
                v2,
                "daily_price_limit",
                "ts_code, trade_date, up_limit, down_limit",
            )
            suspensions = _query(
                v2, "daily_suspension", "ts_code, trade_date, suspend_type"
            )
            basics = _query(
                v2, "daily_basic_pit", "ts_code, trade_date, turnover_rate"
            )

        if bars.empty:
            return pd.DataFrame()
        bars = bars.rename(columns={"vol": "volume"})
        df = bars.merge(adj, on=["ts_code", "trade_date"], how="left")
        df = df.merge(limits, on=["ts_code", "trade_date"], how="left")
        df = df.merge(basics, on=["ts_code", "trade_date"], how="left")
        suspended_codes_dates = set(
            zip(suspensions["ts_code"], suspensions["trade_date"])
        ) if not suspensions.empty else set()
        # p17 §6.1: a daily_suspension record is available in that day's
        # session; presence of the record means the stock did not trade.
        df["suspended"] = [
            (code, date) in suspended_codes_dates
            for code, date in zip(df["ts_code"], df["trade_date"])
        ]
        return df

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

        # Formal price source: the v2 Tushare/PIT tables (p19 §2.1).  The
        # legacy qfq daily table is never read here (legacy-audit only).
        df_raw = self._load_v2_bars(requested_codes, requested_dates)
        if df_raw.empty:
            raise ValueError("No daily bars found")

        # Adjusted-basis derivation consumes RAW adj_factor values and must
        # run before normalize_daily_bars (whose legacy ffill().bfill() is
        # pre-registered as abolished for this path, p19 §2.1).
        scale, adj_degraded = adjusted_price_scale(df_raw)

        df = normalize_daily_bars(df_raw)
        if df.empty:
            raise ValueError("No daily bars found")
        scale = pd.Series(scale, index=df_raw.index).loc[df.index]
        df_adj = apply_adjusted_scale(df, scale)

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

        # Degraded provenance: adjusted-basis anomalies and missing exact
        # constraint records are recorded, never swallowed (p19 §2.1/§2.4).
        self.price_degraded = {}
        for code in adj_degraded:
            self.price_degraded[code] = (
                "adj_factor missing/non-positive on a bar row; adjusted "
                "series fail-closed (p19 §2.1)"
            )
        up_wide = pivot_wide(df, self.ts_codes, self.dates, "up_limit")
        down_wide = pivot_wide(df, self.ts_codes, self.dates, "down_limit")
        missing_limit = ~(np.isfinite(up_wide.to_numpy(dtype=float))
                          & np.isfinite(down_wide.to_numpy(dtype=float)))
        for row, code in enumerate(self.ts_codes):
            if missing_limit[row].any() and code not in self.price_degraded:
                self.price_degraded[code] = (
                    "daily_price_limit record missing for at least one "
                    "loaded session; exact constraint fail-closed (p19 §2.4)"
                )
        if self.price_degraded:
            for code, reason in sorted(self.price_degraded.items()):
                logger.warning(
                    "price semantics degraded for {}: {}", code, reason
                )

        tensor_map: dict[str, torch.Tensor] = {}
        for column in _ZERO_FILLED_COLUMNS:
            wide = pivot_wide(df, self.ts_codes, self.dates, column)
            tensor_map[column] = torch.tensor(
                np.nan_to_num(wide.values, nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            )
        # Adjusted open for the label basis (0-filled like the raw open).
        adj_open_wide = pivot_wide(df_adj, self.ts_codes, self.dates, "open")
        tensor_map["adjusted_open"] = torch.tensor(
            np.nan_to_num(
                adj_open_wide.values, nan=0.0, posinf=0.0, neginf=0.0
            ),
            dtype=torch.float32,
        )
        # Exact constraint matrices: limit prices keep NaN (missing record
        # must stay distinguishable from a 0 limit — fail-closed blocked);
        # suspension is the 0/1 table-driven flag (p19 §2.4).
        tensor_map["up_limit"] = torch.tensor(
            up_wide.to_numpy(dtype=np.float32), dtype=torch.float32
        )
        tensor_map["down_limit"] = torch.tensor(
            down_wide.to_numpy(dtype=np.float32), dtype=torch.float32
        )
        suspended_wide = pivot_wide(
            df.assign(suspended=df["suspended"].astype(float)),
            self.ts_codes,
            self.dates,
            "suspended",
        )
        tensor_map["suspended"] = torch.tensor(
            np.nan_to_num(suspended_wide.to_numpy(dtype=np.float32), nan=0.0),
            dtype=torch.float32,
        )
        self.raw_data_cache = tensor_map

        adjusted_map: dict[str, torch.Tensor] = {}
        for column in ("open", "high", "low", "close", "pre_close"):
            wide = pivot_wide(df_adj, self.ts_codes, self.dates, column)
            # NaN stays NaN: the adjusted cache is the fail-closed view of
            # the adjusted basis — missing is never turned into 0 (p19
            # §2.1); consumers that need the 0-filled dead-bar convention
            # apply it explicitly (raw_data_cache["adjusted_open"]).
            adjusted_map[column] = torch.tensor(
                wide.to_numpy(dtype=np.float32), dtype=torch.float32
            )
        self.adjusted_data_cache = adjusted_map

        # Announce-date-aligned point-in-time fundamentals and capital-flow
        # frames; any failure (e.g. the tables do not exist yet) degrades
        # to neutral frames.
        close_wide = pivot_wide(df, self.ts_codes, self.dates, "close")
        pit = build_pit_frames(self.config, self.ts_codes, self.dates, close_wide)
        capital = build_capital_frames(self.config, self.ts_codes, self.dates)
        apply_cross_sectional_z(
            capital,
            ("MARGIN_BALANCE_TO_CIRC_MV", "MARGIN_BUY_PRESSURE_5"),
        )
        industry_frame = build_industry_member_frame(
            self.config, self.ts_codes, self.dates
        )
        # p17 §12 qualification input for the D7 per-formula gate: the
        # Shenwan index close source may lag at most 3 open sessions; a
        # stale source is recorded as degraded (never silently healthy).
        self.industry_index_lag = industry_index_freshness(
            self.config, self.dates
        )
        if (
            self.industry_index_lag is not None
            and self.industry_index_lag > FRESHNESS_TOLERANCE_SESSIONS
        ):
            self.industry_source_degraded = (
                f"shenwan index close lag {self.industry_index_lag} open "
                "sessions > tolerance (p17 §12)"
            )
            logger.warning(
                "industry source degraded: {}", self.industry_source_degraded
            )

        # p21 §3 (amended 2026-09-08, captain-approved): the E2
        # content-addressed factor-tensor cache.  The identity is derived
        # fail-closed from the store state (``load_cache``); a disabled
        # identity leaves this path byte-identical to pre-wiring
        # behavior, and a hit returns the exact float32 tensor the miss
        # path would have computed (the cast is deterministic).
        self.dataset_id = None
        try:
            with AshareDB(self.config.duckdb_path, read_only=True) as db:
                self.dataset_id = resolve_dataset_id(db, self.config)
        except duckdb.Error as exc:
            logger.warning(
                f"dataset manifest unavailable (legacy database?): {exc}"
            )
            self.dataset_id = None
        cache = factor_cache_for(self.config.data_dir)
        identity = loader_factor_cache_identity(
            resolve_v2_duckdb_path(self.config),
            self.dataset_id,
            self.dates,
            self.ts_codes,
        )
        cached = cache.get(identity) if identity is not None else None
        self.factor_cache_outcome = (
            "disabled" if identity is None else "miss"
        )
        if cached is None:
            # Factors consume the ADJUSTED continuous basis (p19 §2.3);
            # volume, amount and turnover_rate stay raw (nominal shares /
            # yuan).  The exact-limit event family additionally reads the
            # raw nominal prices (close_raw/high_raw/low_raw) and the
            # daily_price_limit columns already merged into df_adj —
            # p17 §5.2 table 5, t22 in-place fix.
            factors = compute_factor_tensor(
                df_adj.assign(
                    close_raw=df["close"], high_raw=df["high"], low_raw=df["low"],
                ),
                self.ts_codes,
                self.dates,
                pit_fundamentals=pit,
                extra_frames=capital,
                industry_frame=industry_frame,
                universe_mask=self.universe_mask,
            )
            self.factor_tensor = torch.tensor(factors, dtype=torch.float32)
            if identity is not None:
                cache.put(
                    identity, np.asarray(factors, dtype=np.float32)
                )
        else:
            self.factor_tensor = torch.tensor(cached, dtype=torch.float32)
            self.factor_cache_outcome = "hit"
        # Dense industry group ids for the VM's CS_NEUTRALIZE operator
        # (unmapped stocks stay NaN so no group is fabricated for them).
        self.industry_codes = torch.tensor(
            encode_industry_frame(industry_frame), dtype=torch.float32
        )

        # Labels (target_ret) consume the ADJUSTED open-to-open series
        # (p19 §2.3): a corporate action between the entry and exit open is
        # an economic return, not a price gap.  Legacy daily/horizon=1
        # adjacent-open field retained for callers outside the P3 research
        # path.  Missing-data masking still prevents suspended / unlisted
        # days from fabricating huge fake returns.
        open_tensor = self.raw_data_cache["adjusted_open"]
        target = self.mask_by_universe(open_to_open_returns(open_tensor.numpy()))
        # Fail-closed adjusted series keep NaN labels (missing never
        # becomes 0 — p19 §2.1); the stock is also untradeable for new
        # entries via tradability_masks().
        if adj_degraded:
            degraded_rows = [
                self.ts_codes.index(code)
                for code in adj_degraded
                if code in self.ts_codes
            ]
            if degraded_rows:
                target[degraded_rows, :] = np.nan
        self.target_ret = torch.tensor(target, dtype=torch.float32)
        # The dataset-manifest binding (and its fail-closed resolution)
        # happens before the factor-cache identity is derived; the tail
        # of load_data only assembles the remaining public tensors.
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

        The exact p19 §2.4 rule over the ``daily_price_limit`` /
        ``daily_suspension`` tables: buy blocked ⟺ open >= up_limit, sell
        blocked ⟺ open <= down_limit, a suspension record blocks both
        sides, and a missing limit record blocks both sides (fail-closed,
        degraded record in :attr:`price_degraded`).  Stocks whose adjusted
        series is fail-closed (adj_factor anomaly) cannot be newly bought
        — their factor/label basis is invalid.  The reward path consumes
        these masks so the training basket can never trade what the engine
        could not have traded; callers slice them to the same date window
        as the signals.
        """

        cached = self._tradability_cache
        if cached is not None:
            return cached
        if not self.raw_data_cache:
            raise ValueError("tradability masks require loaded raw bars")
        raw = self.raw_data_cache
        blocked_buy, _ = exact_tradability_matrix(
            raw["open"].numpy(),
            raw["up_limit"].numpy(),
            raw["down_limit"].numpy(),
            raw["suspended"].numpy(),
            "buy",
        )
        blocked_sell, _ = exact_tradability_matrix(
            raw["open"].numpy(),
            raw["up_limit"].numpy(),
            raw["down_limit"].numpy(),
            raw["suspended"].numpy(),
            "sell",
        )
        if self.universe_mask is None:
            raise ValueError("tradability masks require loaded universe mask")
        # Ineligible stocks cannot be newly bought.  Sell eligibility remains
        # market-driven so an index exit never traps an existing position.
        blocked_buy |= ~self.universe_mask
        # Fail-closed adjusted series: no valid label/factor basis -> no
        # new entries (exits stay market-driven, mirroring the mask rule).
        for code, reason in self.price_degraded.items():
            if "adj_factor" in reason and code in self.ts_codes:
                blocked_buy[self.ts_codes.index(code)] = True
        self._tradability_cache = (blocked_buy, blocked_sell)
        return self._tradability_cache

    def dollar_volume(self) -> np.ndarray:
        """``[stock, date]`` dollar-volume proxy (volume * close, float64).

        The capacity audit (T1-04) consumes this as the ``adv`` argument:
        a position's value divided by its execution-day dollar volume is
        its capacity utilization.  Zero-volume cells (suspensions) carry
        zero and are skipped by the audit.  Raw nominal prices (p19 §2.2).
        """

        if not self.raw_data_cache:
            raise ValueError("dollar volume requires loaded raw bars")
        volume = self.raw_data_cache["volume"].numpy()
        close = self.raw_data_cache["close"].numpy()
        return (volume * close).astype(np.float64)


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
