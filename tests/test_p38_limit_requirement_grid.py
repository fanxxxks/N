"""P38 data-qualification unblock tests (p19 §2.4 requirement grid).

Executable projection of ``docs/p38_data_qualification_unblock_draft.md``
(APPROVED via user direct instruction 2026-09-19 「全部授权」).  The
requirement grid for ``daily_price_limit`` records narrows from the full
(code x loaded-session) grid to exactly the cells that HAVE a bar row
(finite open > 0) — no-bar sessions (pre-listing / post-delisting /
suspension-without-row) are structurally outside the grid, and the
missing definition aligns with the exact tradability matrix (non-finite
or <= 0).  Per-day trading semantics are untouched: N4 pins the exact
matrix behavior so the narrowing can never reach the masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ashare_data.config import ModelConfig
from ashare_data.db import AshareDB

from test_p34_r01_baseline import (
    MINI_CODES,
    _mini_bars,
    _mini_dates,
    build_mini_db,
    mini_env,  # noqa: F401  (pytest fixture, re-exported)
)

#: The code that lists late (bars from session 30 of 130): every earlier
#: loaded session has no bar, no adj factor and no limit row — the
#: pre-listing shape of the 1,630-code production finding.
LATE_CODE = MINI_CODES[3]
LATE_START_INDEX = 30


def _v2_path(data_config) -> Path:
    v2_path = Path(data_config.v2_duckdb_path)
    if not v2_path.is_absolute():
        v2_path = data_config.duckdb_path.parent / v2_path
    return v2_path


def _repopulate_v2_prices(
    data_config,
    *,
    late_start: int | None = LATE_START_INDEX,
    drop_limit_for: set[tuple[str, str]] | None = None,
    limit_overrides: dict[tuple[str, str], tuple[float, float]]
    | None = None,
) -> None:
    """Rebuild the v2 price tables through the conftest authority, with
    the late-listed code's bars (and their limits/adj rows) starting at
    ``late_start``, then re-publish both manifests exactly the way
    ``build_mini_db`` does (no second publication implementation)."""

    from conftest import populate_v2_price_tables, publish_manifest
    from ashare_data.manifest import (
        TUSHARE_V2_CONFIG,
        build_dataset_manifest,
        save_manifest,
    )

    dates = _mini_dates()
    bars = _mini_bars()
    if late_start is not None:
        early = set(dates[:late_start])
        bars = bars[
            ~(
                (bars["ts_code"] == LATE_CODE)
                & (bars["trade_date"].isin(early))
            )
        ]
    populate_v2_price_tables(
        _v2_path(data_config),
        dates,
        MINI_CODES,
        bars,
        drop_limit_for=drop_limit_for,
        limit_overrides=limit_overrides,
    )
    publish_manifest(data_config)
    with AshareDB(_v2_path(data_config)) as v2_rw:
        v2_manifest = build_dataset_manifest(
            v2_rw, TUSHARE_V2_CONFIG, use_cache=False
        )
        save_manifest(v2_rw, TUSHARE_V2_CONFIG, v2_manifest)
    publish_manifest(data_config)


def _loaded_loader(data_config):
    from ashare_model.data_loader import AshareDataLoader

    loader = AshareDataLoader(data_config, ModelConfig())
    loader.confirm_input_snapshot()
    loader.load_data()
    return loader


# --- N1: a late listing with complete limits on its OWN bar sessions is clean --


def test_n1_prelisting_outside_requirement_grid(mini_env):
    """The late-listed code has no bars (and no limits) for the first 30
    sessions; every session it HAS a bar for carries a limit record.
    p38 §2: those cells are structurally outside the requirement grid —
    the loader must report the frames CLEAN.  Pre-narrowing this is the
    1,630-code false disclosure."""

    _, data_config, _ = mini_env
    _repopulate_v2_prices(data_config)
    loader = _loaded_loader(data_config)
    assert LATE_CODE not in loader.price_degraded, (
        f"pre-listing cells must not degrade {LATE_CODE}: "
        f"{loader.price_degraded.get(LATE_CODE)}"
    )
    assert loader.price_degraded == {}
    status = loader.loader_frame_status()
    assert status["status"] == "clean", status["reasons"]


# --- N2: a missing limit ON A BAR SESSION still degrades (the guard) ----------


def test_n2_bar_session_gap_still_degrades(mini_env):
    """The narrowing must never relax the real-gap disclosure: with the
    late listing in place AND one limit row dropped on a session the
    stock HAS a bar for, the stock is degraded (fail-closed, p19 §2.4)."""

    _, data_config, _ = mini_env
    gap_session = _mini_dates()[LATE_START_INDEX + 5]
    _repopulate_v2_prices(
        data_config, drop_limit_for={(LATE_CODE, gap_session)}
    )
    loader = _loaded_loader(data_config)
    assert LATE_CODE in loader.price_degraded
    assert loader.loader_frame_status()["status"] == "degraded"


# --- N3: the missing definition aligns with the exact matrix (<= 0) ----------


def test_n3_nonpositive_limit_is_missing_on_bar_session(mini_env):
    """A full-window code whose bar session carries ``up_limit = 0`` is
    missing a usable limit under the exact matrix's own criterion
    (non-finite OR <= 0) — the disclosure grid must flag it.  The old
    isfinite-only definition let a zero-limit row through."""

    from conftest import populate_v2_price_tables, publish_manifest
    from ashare_data.manifest import (
        TUSHARE_V2_CONFIG,
        build_dataset_manifest,
        save_manifest,
    )

    _, data_config, _ = mini_env
    code = MINI_CODES[0]
    dates = _mini_dates()
    zero_session = dates[40]
    populate_v2_price_tables(
        _v2_path(data_config),
        dates,
        MINI_CODES,
        _mini_bars(),
        limit_overrides={(code, zero_session): (0.0, 1.0)},
    )
    publish_manifest(data_config)
    with AshareDB(_v2_path(data_config)) as v2_rw:
        v2_manifest = build_dataset_manifest(
            v2_rw, TUSHARE_V2_CONFIG, use_cache=False
        )
        save_manifest(v2_rw, TUSHARE_V2_CONFIG, v2_manifest)
    publish_manifest(data_config)

    loader = _loaded_loader(data_config)
    assert code in loader.price_degraded
    assert loader.loader_frame_status()["status"] == "degraded"


# --- N4: the per-day exact semantics are pinned (unreachable by the change) ---


def test_n4_exact_matrix_per_day_semantics_pinned():
    """The narrowing is disclosure-grid only: the exact constraint rule
    keeps blocking both sides on a missing limit (NaN or <= 0) and on a
    dead bar, and keeps the open-vs-limit hits — verbatim p19 §2.4."""

    from ashare_data.processor import _exact_blocked_parts

    o = np.array([[10.0, 10.0, 10.0, 0.0]])
    up = np.array([[11.0, np.nan, 11.0, 11.0]])
    down = np.array([[9.0, 9.0, 0.0, 9.0]])
    sus = np.zeros_like(o)
    # Session 0: clean and inside the band -> tradeable both sides.
    # Session 1: NaN up_limit -> blocked both sides (missing record).
    # Session 2: zero down_limit (<= 0) -> blocked both sides.
    # Session 3: dead bar (open <= 0) -> blocked both sides.
    buy, missing = _exact_blocked_parts(o, up, down, sus, "buy")
    sell, _ = _exact_blocked_parts(o, up, down, sus, "sell")
    assert buy.tolist() == [[False, True, True, True]]
    assert sell.tolist() == [[False, True, True, True]]
    assert missing.tolist() == [[False, True, True, False]]
    # The exact hits stay exact: open at the published limits.
    o2 = np.array([[11.0, 9.0]])
    up2 = np.array([[11.0, 11.0]])
    down2 = np.array([[9.0, 9.0]])
    buy2, _ = _exact_blocked_parts(o2, up2, down2, sus[:, :2], "buy")
    sell2, _ = _exact_blocked_parts(o2, up2, down2, sus[:, :2], "sell")
    assert buy2.tolist() == [[True, False]]
    assert sell2.tolist() == [[False, True]]
