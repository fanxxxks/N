"""t22 RED tests: in-place factor semantic fixes (p17/D4-D6 consumption).

Covers the fix families from the t22 task list, each written against the
DESIRED post-fix behavior (AGENTS §2.2 step 3 — RED before GREEN):

1. `_industry_demean` must consume the per-date PIT membership column of
   the D5 ``industry_membership_pit`` frame (t18 finding: the frozen
   implementation reads only the window-start column, collapsing the
   family to ~3% coverage on the v2 dataset).
2. `INDUSTRY_MOMENTUM` must align the legacy SW index-close namespace
   (bare ``801010``) with the v2 membership namespace (``801010.SI``);
   t18 measured coverage 0.000 from the join mismatch.
3. The limit-event family must seal/break against the exact
   ``daily_price_limit`` prices (p17 §5.2 table 5) instead of the
   per-board ``limit_rate`` approximation + 0.5pp tolerance band, and the
   loader's compute path must carry the raw/limit columns through.
4. The margin frames must read the v2 ``margin_detail_pit`` source with
   p17 §6.1 T+1 availability (pivot by ``available_date``), replacing the
   legacy ``margin_balance`` trade-date pivot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data import config as config_module
from ashare_data.capital_flow import build_capital_frames
from ashare_data.db import AshareDB
from ashare_model.factors import FACTOR_REGISTRY, FactorContext
from ashare_model.vocab import FEATURE_NAMES


def _empty(index, columns):
    return pd.DataFrame(np.nan, index=index, columns=columns)


def _feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


# ---------------------------------------------------------------------------
# 1. Industry demean consumes the per-date PIT membership column.
# ---------------------------------------------------------------------------


def test_industry_demean_uses_per_date_membership():
    """A mid-window industry switch must move the stock into the new
    group's mean from the switch session on."""

    codes = ["A", "B", "C", "D"]
    dates = [f"2024010{i}" for i in range(1, 7)]
    values = [100.0, 10.0, 20.0, 30.0]
    turnover = pd.DataFrame({d: values for d in dates}, index=codes, columns=dates)
    industry = pd.DataFrame(
        {
            "20240101": ["801010.SI", "801010.SI", "801030.SI", "801030.SI"],
            "20240102": ["801010.SI", "801010.SI", "801030.SI", "801030.SI"],
            "20240103": ["801010.SI", "801010.SI", "801030.SI", "801030.SI"],
            # A switches into the 801030 group from 20240104 on.
            "20240104": ["801030.SI", "801010.SI", "801030.SI", "801030.SI"],
            "20240105": ["801030.SI", "801010.SI", "801030.SI", "801030.SI"],
            "20240106": ["801030.SI", "801010.SI", "801030.SI", "801030.SI"],
        },
        index=codes,
        columns=dates,
    )
    ctx = FactorContext(
        ts_codes=list(codes),
        dates=dates,
        close=turnover,
        open=_empty(codes, dates),
        high=_empty(codes, dates),
        low=_empty(codes, dates),
        pre_close=_empty(codes, dates),
        volume=_empty(codes, dates),
        amount=_empty(codes, dates),
        turnover=turnover,
        eligible=pd.DataFrame(True, index=codes, columns=dates),
        industry=industry,
    )
    frame = FACTOR_REGISTRY["IND_REL_TURNOVER"][1](ctx)
    # Pre-switch: A is in 801010 with B -> group mean (100, 10) = 55.
    assert frame.loc["A", "20240103"] == pytest.approx(100.0 - 55.0)
    # Post-switch: A joins 801030 whose members are A(100), C(20), D(30)
    # — A is an eligible member of its own group -> mean 50.
    assert frame.loc["A", "20240106"] == pytest.approx(100.0 - 50.0)


# ---------------------------------------------------------------------------
# 2. Exact daily_price_limit semantics for the limit-event family.
# ---------------------------------------------------------------------------


def _limit_ctx():
    codes = ["600000.SH", "300001.SZ"]
    dates = ["20240102", "20240103", "20240104", "20240105"]
    # 600000: day1 one-word 9.7% vs exact limit 10.98 (no seal — the 0.5pp
    # tolerance band fabricated this event); day2 touched 10.98 / closed
    # 10.50 (break); day3 sealed at 10.98 (event); day4 sealed at the down
    # limit 9.02 (down event).  300001 never seals at the 10% prices.
    raw_close = pd.DataFrame(
        {dates[0]: [10.97, 10.00], dates[1]: [10.50, 10.00],
         dates[2]: [10.98, 10.00], dates[3]: [9.02, 10.00]},
        index=codes,
    )
    raw_high = pd.DataFrame(
        {dates[0]: [10.97, 10.00], dates[1]: [10.98, 10.00],
         dates[2]: [10.98, 10.00], dates[3]: [9.02, 10.00]},
        index=codes,
    )
    raw_low = pd.DataFrame(
        {dates[0]: [10.97, 10.00], dates[1]: [10.40, 10.00],
         dates[2]: [10.98, 10.00], dates[3]: [9.02, 10.00]},
        index=codes,
    )
    up_limit = pd.DataFrame(10.98, index=codes, columns=dates)
    down_limit = pd.DataFrame(9.02, index=codes, columns=dates)
    ctx = FactorContext(
        ts_codes=list(codes),
        dates=dates,
        close=raw_close,
        open=_empty(codes, dates),
        high=raw_high,
        low=raw_low,
        pre_close=_empty(codes, dates),
        volume=_empty(codes, dates),
        amount=_empty(codes, dates),
        turnover=_empty(codes, dates),
        eligible=pd.DataFrame(True, index=codes, columns=dates),
        close_raw=raw_close,
        high_raw=raw_high,
        low_raw=raw_low,
        up_limit=up_limit,
        down_limit=down_limit,
    )
    return ctx, codes, dates


def test_limit_events_use_exact_daily_price_limit():
    ctx, codes, dates = _limit_ctx()
    up = FACTOR_REGISTRY["LIMIT_UP_EVENT"][1](ctx)
    down = FACTOR_REGISTRY["LIMIT_DOWN_EVENT"][1](ctx)
    brk = FACTOR_REGISTRY["LIMIT_BREAK"][1](ctx)
    # Exact semantics: 10.97 != 10.98 -> no seal despite the 9.7% move.
    assert up.loc[codes[0], dates[0]] == 0.0
    # Touched the exact limit price, closed below -> break.
    assert brk.loc[codes[0], dates[1]] == 1.0
    # Sealed at the exact limit price -> event.
    assert up.loc[codes[0], dates[2]] == 1.0
    assert down.loc[codes[0], dates[3]] == 1.0
    # The 20%-board stock never seals at the 10% prices -> all zero.
    assert up.loc[codes[1]].sum() == 0.0
    assert down.loc[codes[1]].sum() == 0.0


def test_limit_events_ride_compute_factor_tensor_wiring():
    """The compute path must carry the raw/limit columns through to the
    event builders (data_loader wires close_raw/high_raw/low_raw)."""

    from ashare_model.factors import compute_factor_tensor

    codes = ["600000.SH", "300001.SZ"]
    dates = ["20240102", "20240103", "20240104", "20240105"]
    rows = []
    for code in codes:
        for d in dates:
            sealed = code == "600000.SH" and d == "20240104"
            close = 10.98 if sealed else 10.00
            rows.append({
                "ts_code": code, "trade_date": d,
                "close": close, "high": close, "low": close, "open": close,
                "pre_close": 10.00, "close_raw": close, "high_raw": close,
                "low_raw": close, "up_limit": 10.98, "down_limit": 9.02,
                "volume": 1.0, "amount": 1.0, "turnover_rate": 1.0,
            })
    bars = pd.DataFrame(rows)
    mask = np.ones((len(codes), len(dates)), dtype=bool)
    tensor = compute_factor_tensor(bars, codes, dates, mask)
    row = tensor[_feature_index("LIMIT_UP_EVENT")]
    # The sealed stock must standardize above the never-sealed stock on
    # the seal day — the exact semantics survive the cross-sectional z.
    assert row[0, 2] > row[1, 2]


# ---------------------------------------------------------------------------
# 3. INDUSTRY_MOMENTUM namespace alignment + 4. margin v2 source.
# ---------------------------------------------------------------------------


def _cfg_with_sw_index(tmp_path: Path, dates: list[str], close: float = 1000.0):
    cfg = config_module.DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        index_rows = [
            {"index_code": "801010", "industry_name": "农业", "trade_date": d,
             "close": close * 1.01**i}
            for i, d in enumerate(dates)
        ]
        db.upsert_sw_index(index_rows, cfg)
    return cfg


def _populate_membership(v2_path: Path, codes: list[str]):
    from tests.conftest import populate_v2_industry_tables

    populate_v2_industry_tables(
        v2_path,
        membership_rows=[
            {"ts_code": code, "index_code": "801010.SI", "level": 1,
             "in_date": "20200101", "out_date": None, "is_new": "0",
             "industry_name": "农业", "available_date": "20200102"}
            for code in codes
        ],
        taxonomy_rows=[
            {"index_code": "801010.SI", "industry_name": "农业", "level": "L1",
             "industry_code": "SW2020-110000", "src": "sw2020"},
        ],
    )


def test_industry_momentum_aligns_legacy_sw_index_namespace(tmp_path: Path):
    cfg = _cfg_with_sw_index(tmp_path, pd.bdate_range("2024-01-01", periods=30).strftime("%Y%m%d").tolist())
    dates = pd.bdate_range("2024-01-01", periods=30).strftime("%Y%m%d").tolist()
    codes = ["000001.SZ", "000505.SZ", "600000.SH"]
    _populate_membership(cfg.duckdb_path.parent / "ashare_v2.duckdb",
                         ["000001.SZ", "000505.SZ"])
    frames = build_capital_frames(cfg, codes, dates)
    industry = frames["INDUSTRY_MOMENTUM"]
    # Members inherit the index 20-day return; the namespace must not
    # collapse the join into all-NaN.
    assert industry.loc["000001.SZ", dates[20]] == pytest.approx(1.01**20 - 1.0)
    assert np.isnan(industry.loc["600000.SH", dates[20]])


def test_margin_frames_read_v2_margin_detail_with_t_plus_1_visibility(tmp_path: Path):
    from tests.conftest import populate_v2_margin_table, populate_v2_industry_tables

    dates = pd.bdate_range("2024-01-01", periods=30).strftime("%Y%m%d").tolist()
    cfg = _cfg_with_sw_index(tmp_path, dates)
    codes = ["000001.SZ", "600000.SH"]
    _populate_membership(cfg.duckdb_path.parent / "ashare_v2.duckdb", ["000001.SZ"])
    # Margin details of trade_date dates[k] become visible at dates[k+1]
    # (p17 §6.1 T+1); the balance grows by 1.0 per detail.
    populate_v2_margin_table(
        cfg.duckdb_path.parent / "ashare_v2.duckdb",
        rows=[
            {"ts_code": "000001.SZ", "trade_date": dates[k],
             "available_date": dates[k + 1], "rzye": 100.0 + (k - 3)}
            for k in range(3, 26)
        ],
    )
    frames = build_capital_frames(cfg, codes, dates)
    margin = frames["MARGIN_BALANCE_CHG"]
    # 20-session change at column t = wide[t] / wide[t-20] - 1 where
    # wide[t] is the detail VISIBLE at t (available_date = t), i.e. the
    # trade_date dates[t-1] row (rzye = 100 + (t-1-3)).  A trade-date
    # pivot would read the dates[t] row instead and disagree.
    expected_24 = (100.0 + (23 - 3)) / 100.0 - 1.0
    assert margin.loc["000001.SZ", dates[24]] == pytest.approx(expected_24)
    # The never-margin stock stays NaN.
    assert np.isnan(margin.loc["600000.SH", dates[24]])


def test_t70_margin_balance_chg_direction_reregistered_negative():
    """t70 (p20 draft, captain+governance double-signed): the authored +1
    of MARGIN_BALANCE_CHG was falsified in four independent audit rounds
    (t2/v4 legacy, t18/t22 v2 — IC10 -0.0263..-0.0335, ICIR -0.237..-0.414,
    annual direction stability <=0.08) — the direction re-preregisters to
    -1 (leveraged-inflow crowding economics, four-round-consistent with
    the MARGIN_CROWD_60 -1 family member).  The factor formula, source and
    PIT visibility are unchanged (metadata-only flip)."""

    from ashare_model.feature_metadata import FEATURE_METADATA

    meta = FEATURE_METADATA["MARGIN_BALANCE_CHG"]
    assert meta.expected_direction == -1
    assert meta.availability_rule.startswith("v2 margin_detail_pit")
    # A04 (p28 I4, whitelist §10.1 case 2 — the approved p28 contract
    # unified the margin family's depends_on convention): the pin moves
    # from the bare table name to the table.field form — a STRONGER
    # assertion (the consumed field is named and auditable, p17 §5.2).
    assert meta.depends_on == ("margin_detail_pit.rzye",)


# ---------------------------------------------------------------------------
# 5. p9 Part A execution (governance ruling on the keep-list conflict):
#    TURNOVER_MA20 deprecated — exits the sampling space, keeps its token
#    and computation, loses promotion eligibility.
# ---------------------------------------------------------------------------


def test_turnover_ma20_deprecated_per_part_a_ruling():
    from ashare_model.feature_metadata import FEATURE_METADATA
    from ashare_model.vocab import (
        DEPRECATION_REASONS,
        DEPRECATED_FEATURE_NAMES,
        FEATURE_NAMES,
        FORMULA_VOCAB,
        GRAMMAR_VERSION,
    )

    # Part A: |rho| = 0.914 stable across four audit rounds vs the stronger
    # TURNOVER_STD20 representative (governance ruling, option A).
    assert "TURNOVER_MA20" in DEPRECATION_REASONS
    assert "TURNOVER_STD20" in DEPRECATION_REASONS or True  # STD20 stays live
    assert "TURNOVER_STD20" not in DEPRECATION_REASONS
    # Deprecation never removes a name: the token space and the computation
    # survive (legacy formulas keep resolving by name).
    assert "TURNOVER_MA20" in FEATURE_NAMES
    assert "TURNOVER_MA20" in FORMULA_VOCAB.feature_names
    # The deprecation left the sampling space -> the grammar generation
    # bumped (the same discipline as every prior deprecated-set change).
    # The p23 v4 F1 append subsequently bumped 7 -> 8 (the append keeps
    # ids 1-77 stable, so this test's assertions are unchanged).
    assert GRAMMAR_VERSION == 9  # v9 (p24, the user-approved F2 batch) on top of v7 (this deprecation) + the p23 v5 round-trip
    # Promotion eligibility lost (the registry mirrors the metadata).
    assert FEATURE_METADATA["TURNOVER_MA20"].promotion_allowed is False


def test_industry_momentum_flip_reverted_per_final_arbitration():
    """Captain final arbitration (2026-09-07): the -1 flip (12a7ed7)
    executed a REJECTED proposal — the redundancy pre-check failure (the
    corrected dIC -0.0167 below every IND_REL_* family member's increment)
    stands, and the direction confirmation (the -1-aligned IC positive at
    every horizon) does not cure the increment deficiency.  The revert
    restores the authored +1 direction (the pre-registered-hypothesis
    history semantics) with the promotion block maintained; BOTH findings
    are preserved in the metadata annotation (the authored +1 falsified in
    two clean rounds AND the -1 reversal observed-but-rejected)."""

    from ashare_model.feature_metadata import FEATURE_METADATA

    meta = FEATURE_METADATA["INDUSTRY_MOMENTUM"]
    # The authored +1 direction restored (the falsified-evidence semantics).
    assert meta.expected_direction == 1
    # The promotion block stands (the negative-result disposition).
    assert meta.promotion_allowed is False


def test_industry_momentum_negative_result_promotion_blocked():
    """The t24-routing negative-result disposition (promotion eligibility
    withdrawn) SURVIVES the flip revert — the durable invariant across
    every arbitration stage (the rejection, the flip, and the revert)."""
    from ashare_model.feature_metadata import FEATURE_METADATA

    meta = FEATURE_METADATA["INDUSTRY_MOMENTUM"]
    assert meta.promotion_allowed is False
