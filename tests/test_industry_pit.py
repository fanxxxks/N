"""D5 (p17 申万 PIT) contracts: industry membership intervals.

Executes the D5 work-package acceptance list as executable projections of
the approved p17 contract (industry_membership_pit, 勘误 #2 expanded rows)
and the t14 task description:

* **Interval semantics** — a stock's industry is valid on
  ``[in_date, out_date)``; no industry before inclusion, none on the
  out_date itself; out_date NULL means still a member (open-ended).
* **PIT visibility** — the assignment becomes visible only from
  ``available_date`` (p17 §6.1: in_date 的下一开市会话), never on the
  inclusion day itself.
* **Uniqueness fail-closed** — the same stock at the same level has at
  most one valid industry per day (p17 §14.3/§6.3); an overlapping
  interval pair refuses to build (fail-closed, never a silent merge).
* **Level scoping** — the default frame is Shenwan L1; L2/L3 rows never
  leak into it, and an explicit level request returns that level only.
* **Generation traceability** — membership rows carry ``is_new``
  (2021 taxonomy switch); the frame's index codes resolve through
  ``industry_taxonomy`` so the generation of every assignment is
  traceable.
* **Legacy retirement** — the current-snapshot ``sw_industry_member``
  table is out of the formal computation: snapshot writes never rewrite
  the built frame (append-only proof), and the frame is derived purely
  from the PIT intervals.
* **INDUSTRY_MOMENTUM** — consumes the per-date PIT membership (a stock
  switching industries gets the new industry's return series after the
  switch).
* **Freshness gate** — the Shenwan index source's open-session lag
  (p17 §12: ≤ 3) is measurable and exposed for the D7 qualification
  gate.
* **Tier discipline** — the C→B upgrade requires the pre-registered
  gate (历史覆盖 / 官方公告对账 / 可用时间); until it fully passes the
  industry family stays SNAPSHOT / Tier C (p17 §7: 未验证维持 C) — the
  gate computation is machine-checkable and the un-verifiable
  sub-gate (官方公告对账) is recorded as the gap, never silently
  upgraded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.capital_flow import (
    build_capital_frames,
    build_industry_member_frame,
    industry_index_freshness,
    industry_membership_gate,
    load_industry_taxonomy,
)
from ashare_data.config import DataConfig
from ashare_data.db import AshareDB
from ashare_model.feature_registry import PitLevel, pit_level_of
from tests.conftest import populate_v2_industry_tables, publish_manifest

DATES = [f"2024010{d}" for d in range(1, 9)]  # 8 open sessions


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


def _membership(rows: list[dict]) -> list[dict]:
    """Fill the canonical expanded-row shape with defaults per test."""

    base = {"level": 1, "is_new": "0", "industry_name": "旧一代"}
    return [{**base, **row} for row in rows]


def _taxonomy(rows: list[dict]) -> list[dict]:
    base = {"level": "L1", "src": "sw2021"}
    return [{**base, **row} for row in rows]


# --- interval semantics + PIT visibility -------------------------------------


def test_interval_semantics_and_pit_visibility(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240102",
                    "out_date": "20240105",
                    "available_date": "20240103",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801020",
                    "in_date": "20240105",
                    "out_date": None,
                    "is_new": "1",
                    "industry_name": "新一代",
                    "available_date": "20240106",
                },
            ]
        ),
    )
    frame = build_industry_member_frame(config, ["000001.SZ"], DATES)
    # Before inclusion (and before availability): no industry.
    assert pd.isna(frame.loc["000001.SZ", "20240101"])
    assert pd.isna(frame.loc["000001.SZ", "20240102"])  # in_date itself is
    # not yet visible: available_date = 20240103 (p17 §6.1).
    assert frame.loc["000001.SZ", "20240103"] == "801010"
    assert frame.loc["000001.SZ", "20240104"] == "801010"
    # The out_date itself no longer belongs to the old industry, and the
    # same-day successor is not yet visible (available 20240106).
    assert pd.isna(frame.loc["000001.SZ", "20240105"])
    assert frame.loc["000001.SZ", "20240106"] == "801020"
    assert frame.loc["000001.SZ", "20240107"] == "801020"


def test_open_ended_membership_runs_to_window_end(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "600000.SH",
                    "index_code": "801010",
                    "in_date": "20240101",
                    "out_date": None,
                    "available_date": "20240102",
                }
            ]
        ),
    )
    frame = build_industry_member_frame(config, ["600000.SH"], DATES)
    # out_date NULL = still a member (p17 §5.2 table 8); visible from the
    # available session (p17 §6.1: in_date 的次一开市会话).
    assert pd.isna(frame.loc["600000.SH", "20240101"])
    assert (frame.loc["600000.SH", "20240102":] == "801010").all()


# --- uniqueness fail-closed (p17 §14.3/§6.3) ----------------------------------


def test_overlapping_same_level_intervals_fail_closed(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240102",
                    "out_date": "20240105",
                    "available_date": "20240103",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801020",
                    "in_date": "20240104",
                    "out_date": None,
                    "available_date": "20240105",
                },
            ]
        ),
    )
    # 20240104 has two valid L1 industries for one stock: fail-closed.
    with pytest.raises(Exception, match="overlapping|one valid industry"):
        build_industry_member_frame(config, ["000001.SZ"], DATES)


def test_adjacent_intervals_are_not_overlap(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240102",
                    "out_date": "20240105",
                    "available_date": "20240103",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801020",
                    "in_date": "20240105",
                    "out_date": None,
                    "available_date": "20240106",
                },
            ]
        ),
    )
    frame = build_industry_member_frame(config, ["000001.SZ"], DATES)
    assert frame.loc["000001.SZ", "20240104"] == "801010"
    assert frame.loc["000001.SZ", "20240106"] == "801020"


# --- level scoping -------------------------------------------------------------


def test_level_filter_scopes_the_frame(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "level": 1,
                    "in_date": "20240101",
                    "out_date": None,
                    "available_date": "20240102",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801011",
                    "level": 2,
                    "in_date": "20240101",
                    "out_date": None,
                    "is_new": "1",
                    "industry_name": "二级",
                    "available_date": "20240101",
                },
            ]
        ),
    )
    l1 = build_industry_member_frame(config, ["000001.SZ"], DATES)
    assert pd.isna(l1.loc["000001.SZ", "20240101"])
    assert (l1.loc["000001.SZ", "20240102":] == "801010").all()  # L2 never leaks into L1
    l2 = build_industry_member_frame(config, ["000001.SZ"], DATES, level=2)
    assert (l2.loc["000001.SZ", "20240102":] == "801011").all()


def test_overlap_is_scoped_per_level(tmp_path: Path):
    """Two L2 intervals may overlap in time without violating the L1
    uniqueness rule — and vice versa (the uniqueness scope is
    (ts_code, level, day), p17 §14.3)."""

    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "level": 1,
                    "in_date": "20240101",
                    "out_date": None,
                    "available_date": "20240101",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801011",
                    "level": 2,
                    "in_date": "20240101",
                    "out_date": "20240104",
                    "available_date": "20240101",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801012",
                    "level": 2,
                    "in_date": "20240103",
                    "out_date": None,
                    "available_date": "20240103",
                },
            ]
        ),
    )
    # L1 fine; the L2 overlap (801011 x 801012 on 01/03) fails closed.
    build_industry_member_frame(config, ["000001.SZ"], DATES)
    with pytest.raises(Exception, match="overlapping|one valid industry"):
        build_industry_member_frame(config, ["000001.SZ"], DATES, level=2)


# --- generation traceability ----------------------------------------------------


def test_taxonomy_generations_resolve_frame_codes(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240101",
                    "out_date": "20240104",
                    "available_date": "20240102",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801030",
                    "in_date": "20240104",
                    "out_date": None,
                    "is_new": "1",
                    "industry_name": "新一代",
                    "available_date": "20240105",
                },
            ]
        ),
        taxonomy_rows=_taxonomy(
            [
                {
                    "index_code": "801010",
                    "industry_name": "旧一代",
                    "industry_code": "SW2020-110000",
                    "src": "sw2020",
                },
                {
                    "index_code": "801030",
                    "industry_name": "新一代",
                    "industry_code": "SW2021-110000",
                    "src": "sw2021",
                },
            ]
        ),
    )
    frame = build_industry_member_frame(config, ["000001.SZ"], DATES)
    taxonomy = load_industry_taxonomy(config)
    # Every assigned code resolves to exactly one generation (2021 switch
    # traceable through industry_taxonomy).
    assert taxonomy["index_code"].is_unique
    generation = taxonomy.set_index("index_code")["src"]
    assert generation["801010"] == "sw2020"
    assert generation["801030"] == "sw2021"
    # The frame itself switches with the membership intervals: the old
    # generation ends at 20240104 and the new one becomes visible on
    # 20240105 (its available_date, p17 §6.1).
    assert frame.loc["000001.SZ", "20240103"] == "801010"
    assert pd.isna(frame.loc["000001.SZ", "20240104"])
    assert frame.loc["000001.SZ", "20240105"] == "801030"


# --- legacy snapshot retirement (append-only proof) -----------------------------


def test_current_snapshot_writes_never_rewrite_the_frame(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240101",
                    "out_date": "20240104",
                    "available_date": "20240102",
                }
            ]
        ),
    )
    before = build_industry_member_frame(config, ["000001.SZ"], DATES)
    # A "current snapshot" change (today's SW membership says the stock is
    # in a different industry since forever) must not rewrite history.
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {config.sw_member_table} "
            "(index_code VARCHAR, ts_code VARCHAR, PRIMARY KEY (index_code, ts_code))"
        )
        db.execute(
            f"INSERT INTO {config.sw_member_table} VALUES ('801020', '000001.SZ')"
        )
    after = build_industry_member_frame(config, ["000001.SZ"], DATES)
    pd.testing.assert_frame_equal(before, after)
    # And the interval end still governs: no industry after out_date.
    assert pd.isna(after.loc["000001.SZ", "20240104"])


def test_missing_industry_source_degrades_to_nan(tmp_path: Path):
    config = _config(tmp_path)
    # No v2 industry tables at all: the Tier C auxiliary source degrades to
    # the all-NaN frame (existing loader semantics, never fabricated).
    frame = build_industry_member_frame(config, ["000001.SZ"], DATES)
    assert frame.isna().all().all()


# --- INDUSTRY_MOMENTUM consumes per-date PIT membership -------------------------


def test_industry_momentum_follows_pit_membership_switch(tmp_path: Path):
    n_dates = 25  # INDUSTRY_MOMENTUM uses a 20-day index return: finite
    # from the 21st session, so the switch contrast is observable.
    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y%m%d").tolist()
    switch = 22  # membership switch session index: columns 20-21 stay
    # pre-switch (801010) so both return series are observable post-warmup.
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": dates[0],
                    "out_date": dates[switch],
                    "available_date": dates[1],
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801020",
                    "in_date": dates[switch],
                    "out_date": None,
                    "is_new": "1",
                    "industry_name": "新一代",
                    "available_date": dates[switch + 1],
                },
            ]
        ),
    )
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {config.sw_index_table} "
            "(index_code VARCHAR, industry_name VARCHAR, trade_date VARCHAR, "
            "close DOUBLE, PRIMARY KEY (index_code, trade_date))"
        )
        rows = []
        for date_index, date in enumerate(dates):
            # 801010 rises 1%/day; 801020 falls 1%/day — perfectly
            # distinguishable series.
            rows.append(
                {
                    "index_code": "801010",
                    "industry_name": "旧一代",
                    "trade_date": date,
                    "close": 100.0 * (1.01**date_index),
                }
            )
            rows.append(
                {
                    "index_code": "801020",
                    "industry_name": "新一代",
                    "trade_date": date,
                    "close": 100.0 * (0.99**date_index),
                }
            )
        db.register("_sw_index_rows", pd.DataFrame(rows))
        db.execute(
            f"INSERT INTO {config.sw_index_table} "
            "SELECT index_code, industry_name, trade_date, close "
            "FROM _sw_index_rows"
        )
    frames = build_capital_frames(config, ["000001.SZ"], dates)
    momentum = frames["INDUSTRY_MOMENTUM"].loc["000001.SZ"]
    # While a member of 801010 the stock carries 801010's 20-day return;
    # after the PIT switch (session 22) it carries 801020's (opposite
    # direction).  Both are constant geometric series, so the values are
    # exactly 1.01^20-1 / 0.99^20-1.
    assert momentum.loc[dates[20]] == pytest.approx(1.01**20 - 1.0, rel=1e-9)
    assert momentum.loc[dates[21]] == pytest.approx(1.01**20 - 1.0, rel=1e-9)
    # The successor's available_date is the session AFTER the switch, so
    # the switch session itself carries no industry (NaN), then 801020.
    assert pd.isna(momentum.loc[dates[switch]])
    for date in dates[switch + 1:]:
        assert momentum.loc[date] == pytest.approx(0.99**20 - 1.0, rel=1e-9)


# --- freshness gate (p17 §12: 申万指数侧数据滞后 ≤ 3 会话) -----------------------


def test_industry_index_freshness_measures_open_session_lag(tmp_path: Path):
    config = _config(tmp_path)
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {config.sw_index_table} "
            "(index_code VARCHAR, industry_name VARCHAR, trade_date VARCHAR, "
            "close DOUBLE, PRIMARY KEY (index_code, trade_date))"
        )
        rows = [
            {
                "index_code": "801010",
                "industry_name": "旧一代",
                "trade_date": date,
                "close": 100.0,
            }
            for date in DATES[:-3]  # latest close is 3 sessions before the end
        ]
        db.register("_sw_index_rows", pd.DataFrame(rows))
        db.execute(
            f"INSERT INTO {config.sw_index_table} "
            "SELECT index_code, industry_name, trade_date, close "
            "FROM _sw_index_rows"
        )
    lag = industry_index_freshness(config, DATES)
    assert lag == 3  # three open sessions strictly after the latest close
    assert lag <= 3  # within the p17 §12 tolerance

    stale = industry_index_freshness(config, DATES, tolerance=2)
    assert stale > 2  # the same source fails the tighter tolerance


def test_industry_index_freshness_without_source_is_none(tmp_path: Path):
    config = _config(tmp_path)
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
    assert industry_index_freshness(config, DATES) is None


# --- membership qualification gate (p17 §12 + §7 C→B thresholds) -----------------


def test_membership_gate_computes_coverage_overlap_and_availability(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240101",
                    "out_date": "20240105",
                    "available_date": "20240102",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801020",
                    "in_date": "20240105",
                    "out_date": None,
                    "is_new": "1",
                    "available_date": "20240105",
                },
                {
                    "ts_code": "600000.SH",
                    "index_code": "801010",
                    "in_date": "20240101",
                    "out_date": None,
                    "available_date": "20240102",
                },
            ]
        ),
    )
    gate = industry_membership_gate(config, DATES)
    # 000001.SZ: span = 01/01..01/08 (8 sessions), PIT-visible = 7 (01/01
    # is before its available_date 01/02); 600000.SH visible = 7 of 8.
    assert gate["coverage"] == pytest.approx((7 + 7) / 16)
    assert gate["overlap_violations"] == 0
    assert gate["availability_complete"] is True
    # The official-announcement reconciliation sub-gate is NOT computable
    # in-repo: the gate result must carry the open gap honestly.
    assert gate["official_reconciliation"] == "not_verifiable_in_repo"


def test_membership_gate_flags_overlap(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_industry_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=_membership(
            [
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801010",
                    "in_date": "20240101",
                    "out_date": "20240105",
                    "available_date": "20240102",
                },
                {
                    "ts_code": "000001.SZ",
                    "index_code": "801020",
                    "in_date": "20240104",
                    "out_date": None,
                    "available_date": "20240105",
                },
            ]
        ),
    )
    gate = industry_membership_gate(config, DATES)
    assert gate["overlap_violations"] == 1


# --- tier discipline: no premature C→B upgrade -----------------------------------


def test_industry_family_stays_tier_c_pending_gate(tmp_path: Path):
    """p17 §7: the membership source upgrades to B only after the D5
    verification gate fully passes (历史覆盖 / 官方公告对账 / 可用时间).
    The official-reconciliation sub-gate is not verifiable in-repo, so the
    industry family stays SNAPSHOT / Tier C — locked here so no later
    change upgrades it silently."""

    from ashare_model.data_tier import DataTier, feature_tier

    for name in (
        "IND_REL_RET_5",
        "IND_REL_RET_20",
        "IND_REL_RET_60",
        "IND_REL_RET_120",
        "IND_REL_VOL_20",
        "IND_REL_TURNOVER",
        "INDUSTRY_MOMENTUM",
    ):
        assert pit_level_of(name) is PitLevel.SNAPSHOT
        assert feature_tier(name) is DataTier.C


# --- loader freshness exposure (p17 §12 qualification input) --------------------


def test_loader_exposes_industry_index_freshness(populated_db):
    """The loader records the Shenwan index lag and degrades explicitly
    beyond the p17 §12 tolerance — the D7 qualification gate consumes
    this record instead of re-deriving it."""

    from ashare_model.data_loader import AshareDataLoader
    from ashare_data.config import ModelConfig

    with AshareDB(populated_db.duckdb_path) as db:
        db.create_schema(populated_db)
        dates = pd.bdate_range("2024-01-01", periods=40).strftime("%Y%m%d").tolist()
        db.register(
            "_sw_index_rows",
            pd.DataFrame(
                [
                    {
                        "index_code": "801010",
                        "industry_name": "旧一代",
                        "trade_date": date,
                        "close": 100.0,
                    }
                    for date in dates[:-5]  # lag = 5 open sessions > 3
                ]
            ),
        )
        db.execute(
            f"INSERT INTO {populated_db.sw_index_table} "
            "SELECT index_code, industry_name, trade_date, close "
            "FROM _sw_index_rows"
        )
    # A02 OB-1 (CLOSED): this test fabricates a stale Shenwan index by
    # rewriting the manifest-covered `sw_industry_index` table after the
    # fixture published its composite generation.  Before the OB-1 fix,
    # republishing could not make the database confirmable (a same-second
    # stale manifest row won the `created_at DESC, dataset_id DESC`
    # tiebreak, and `ON CONFLICT DO NOTHING` made the wrong selection
    # permanent).  Publication generations fixed that ordering, so the
    # formal (default) mode is restored here: republish the composite
    # generation over the rewritten content and confirm it (see
    # tests/test_manifest.py::test_a02_republishing_restores_confirmability
    # for the OB-1 regression itself).
    publish_manifest(populated_db)
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    assert loader.industry_index_lag == 5
    assert loader.industry_source_degraded is not None
    assert "p17 §12" in loader.industry_source_degraded
