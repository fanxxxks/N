"""A-share cross-sectional factor computation.

Every factor is registered in :data:`FACTOR_REGISTRY` with declarative
metadata (family, required daily-bar columns, warmup, description) and a
pure function over the shared :class:`FactorContext`.  The engine pivots
exactly the columns the requested factors need, computes each registered
factor once, then cross-sectionally standardizes the stack.  Features with
no local data source are declared separately (point-in-time fundamentals
from :mod:`ashare_data.fundamentals`, external capital-flow sources) and
stay neutral instead of being fabricated.

Adding a factor therefore means adding one registry entry plus tests; the
metadata drives diagnostics and family-level ablation experiments.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ashare_data.processor import (
    limit_rate,
    normalize_daily_bars,
    pivot_wide,
    winsorize_cross_section,
)

from .vocab import FEATURE_NAMES

# Features with a local data source outside the daily bars: the PIT
# fundamentals and the capital-flow factors are injected as pre-built
# frames by the loader.  Only NORTHBOUND_CHG has no free historical feed
# (daily disclosure stopped in Aug 2024) and stays neutral.

# FACTOR_COMPUTE_VERSION stays the version owner here (the version-owner
# index pins ashare_model.factors; test_runspec asserts the import path).
# D5 industry-membership PIT + D6 financial version-table as-of both
# change factor values for affected stocks — no historical migration.
FACTOR_COMPUTE_VERSION = 3

# --- Mechanical split (t79) ------------------------------------------------
# The formula layer (FactorSpec/FactorContext, primitives, the registry)
# moved to ``factor_formulas.py``; this module keeps the engine and
# re-exports the moved names (import-surface compatibility, guarded by
# tests/test_factors.py facade identity test).
from .factor_formulas import (  # noqa: F401
    BAR_COLUMNS,
    FAMILIES,
    FACTOR_REGISTRY,
    FactorContext,
    FactorSpec,
    NEUTRAL_FEATURE_NAMES,
    SPARSE_EVENT_FEATURES,
    _factor_amount_share,
    _factor_atr,
    _factor_capm,
    _factor_macd,
    _factor_macd_cached,
    _factor_rsi,
    _limit_break,
    _limit_down_streak,
    _limit_events,
    _limit_event_frame,
    _limit_rate_per_stock,
    _limit_streak,
    _limit_up_count,
    _make_registry,
    _returns,
    _rolling_capm,
    _rolling_corr,
    _rolling_kurt,
    _rolling_max,
    _rolling_mean,
    _rolling_std,
    _safe_ratio,
    _shift_ratio,
    _sparse_safe_standardize,
    _industry_demean,
    _amount_share_cached,
    _turnover_smoothed,
    _turnover_vol,
    _crowd_ratio,
    _crowd_turnover,
    _ewm_mean,
    _streak_frame,
)


class AshareFactorEngine:
    """Compute standardized price-volume and fundamental factor tensors."""

    def __init__(self, feature_names: list[str] | tuple[str, ...] | None = None):
        self.feature_names = list(feature_names or FEATURE_NAMES)

    @staticmethod
    def _pivot(
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        value_col: str,
    ) -> pd.DataFrame:
        return pivot_wide(bars, ts_codes, dates, value_col)

    def _build_context(
        self,
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        close: pd.DataFrame,
        eligible_frame: pd.DataFrame,
        industry_frame: pd.DataFrame | None = None,
    ) -> FactorContext:
        """Pivot exactly the columns the requested registered factors need."""

        needed: set[str] = set()
        for name in self.feature_names:
            entry = FACTOR_REGISTRY.get(name)
            if entry is not None:
                needed.update(entry[0].required_columns)
        empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        frames: dict[str, pd.DataFrame] = {col: empty.copy() for col in BAR_COLUMNS}
        for col in needed:
            frames[col] = self._pivot(bars, ts_codes, dates, col)
        return FactorContext(
            ts_codes=list(ts_codes),
            dates=list(dates),
            close=frames["close"],
            open=frames["open"],
            high=frames["high"],
            low=frames["low"],
            pre_close=frames["pre_close"],
            volume=frames["volume"],
            amount=frames["amount"],
            turnover=frames["turnover_rate"],
            industry=industry_frame,
            eligible=eligible_frame,
        )

    def compute_factor_tensor(
        self,
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        universe_mask: np.ndarray,
        pit_fundamentals: dict[str, pd.DataFrame] | None = None,
        extra_frames: dict[str, pd.DataFrame] | None = None,
        industry_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Return a ``[feature, stock, date]`` float32 tensor.

        ``pit_fundamentals`` maps a PIT feature name to its
        disclosure-aligned ``[stock x date]`` frame (built by
        :func:`ashare_data.fundamentals.build_pit_frames`); ``extra_frames``
        carries the other externally fed features (margin/industry, built by
        :func:`ashare_data.capital_flow.build_capital_frames`);
        ``industry_frame`` is the ``[stock x date]`` Shenwan first-level
        industry code frame feeding the industry-relative factors (built by
        :func:`ashare_data.capital_flow.build_industry_member_frame`).
        Frames are injected verbatim; the no-lookahead guarantees live in
        the builders.  Missing names/frames stay neutral (0) after
        standardization.

        ``universe_mask`` is the mandatory ``[stock x date]`` bool PIT
        eligibility mask aligned with ``(ts_codes, dates)``.  Every
        stock-axis reference statistic (winsorize quantiles / median / MAD,
        the CAPM equal-weight market return, the amount-share denominator,
        the industry means) is computed only over eligible & finite cells,
        so stocks outside the universe cannot change eligible stocks'
        factor values.  A stock's own bar history is never masked: e.g. the
        join-day momentum still reads its pre-join prices.  A mask whose
        shape does not match ``(ts_codes, dates)`` raises ``ValueError``.
        """

        bars = normalize_daily_bars(bars)
        date_order = sorted(range(len(dates)), key=dates.__getitem__)
        dates = [dates[i] for i in date_order]
        universe_mask = np.asarray(universe_mask, dtype=bool)
        if universe_mask.shape != (len(ts_codes), len(dates)):
            raise ValueError(
                f"universe_mask shape {universe_mask.shape} does not match "
                f"(stock={len(ts_codes)}, date={len(dates)})"
            )
        universe_mask = universe_mask[:, date_order]
        eligible_frame = pd.DataFrame(
            universe_mask, index=ts_codes, columns=dates
        )
        close = self._pivot(bars, ts_codes, dates, "close")
        context = self._build_context(
            bars, ts_codes, dates, close, eligible_frame, industry_frame
        )

        empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        raw_factors: dict[str, pd.DataFrame] = {}
        for name in self.feature_names:
            entry = FACTOR_REGISTRY.get(name)
            raw_factors[name] = entry[1](context) if entry is not None else empty.copy()

        # Point-in-time fundamentals and capital-flow factors: inject the
        # pre-built frames; everything else stays neutral.
        injected: dict[str, pd.DataFrame] = {}
        injected.update(pit_fundamentals or {})
        injected.update(extra_frames or {})
        for name, frame in injected.items():
            if name not in self.feature_names:
                continue
            if frame is not None and not frame.empty:
                raw_factors[name] = frame

        standardized = []
        for name in self.feature_names:
            frame = raw_factors.get(name, empty)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame().T
            frame = frame.reindex(index=ts_codes, columns=dates)
            if name in SPARSE_EVENT_FEATURES:
                # P9 §5.3: sparse event features skip the quantile
                # winsorize (t2 finding F1) and standardize on the plain
                # mean/std of the eligible cross-section with a floored
                # standard deviation.
                frame = _sparse_safe_standardize(frame, eligible=universe_mask)
            else:
                frame = winsorize_cross_section(frame, eligible=universe_mask)
            # After cross-sectional standardization 0 is the neutral value.
            standardized.append(frame.fillna(0.0).values.astype(np.float32))

        return np.stack(standardized, axis=0)


def compute_factor_tensor(
    bars: pd.DataFrame,
    ts_codes: list[str],
    dates: list[str],
    universe_mask: np.ndarray,
    pit_fundamentals: dict[str, pd.DataFrame] | None = None,
    extra_frames: dict[str, pd.DataFrame] | None = None,
    industry_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    return AshareFactorEngine().compute_factor_tensor(
        bars, ts_codes, dates, universe_mask, pit_fundamentals, extra_frames,
        industry_frame,
    )


def ablate_factors(
    tensor: np.ndarray,
    excluded_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Zero the rows of the excluded features (family ablation).

    The tensor keeps its full ``[feature, stock, date]`` shape so formula
    token ids stay valid; the excluded features simply become neutral (0).
    """

    out = np.asarray(tensor, dtype=np.float32).copy()
    for name in excluded_names:
        if name in FEATURE_NAMES:
            out[FEATURE_NAMES.index(name)] = 0.0
    return out
