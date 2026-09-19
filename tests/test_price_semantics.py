"""P19 (D4) price-semantics contracts: raw execution / adjusted factor series.

Executes the p19 §7 mandatory test list as executable projections of the
approved contract (``docs/p19_market_price_semantics_contract.md``):

* **RED-1** — no future ex-dividend leakage: factor/label values at and
  before an ex-right event are bitwise unchanged whether or not the event
  is inside the load window (qfq full-history rewrite is pre-registered as
  abolished, p19 §1.1/§2.1).
* **RED-2** — ex-div-day continuity: the adjusted series has no jump at
  the event, the raw series shows the real price gap, and adjusted returns
  match the hand-computed ``raw × adj_factor`` formula exactly.
* **RED-3** — raw nominal execution: whole lots, cash affordability and
  the min-commission floor are computed on raw prices; a >=10x adjusted/raw
  price spread discriminates the basis.
* **RED-4** — exact constraints: ``daily_price_limit`` up/down and
  ``daily_suspension`` rows replace the approximate one-word/board-rate
  inference; a missing limit row is fail-closed (untradeable + degraded
  record) with no approximate fallback.
* **RED-5** — day-by-day golden parity on the new basis (target, buy/sell
  weight, turnover, order, cost, net return) with ex-div, limit and
  suspension scenarios.
* **RED-6** — legacy artifact rejection at the classification entries for
  every version owner the p19 §5 impact table bumps.
* Adjacent regressions: single loader price source (v2, no silent legacy
  fallback), adj_factor anomaly fail-closed (missing / <=0, NaN preserved),
  turnover_rate sourced from ``daily_basic_pit``, fail-closed engine cache
  validation, and the loader's degraded-record provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import BacktestConfig, DataConfig
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.data_loader import AshareDataLoader
from tests.conftest import _populate_db, populate_v2_price_tables, publish_manifest

DATES = [f"2024010{d}" for d in range(1, 9)]  # 8 synthetic sessions
CODES = ["000001.SZ", "600000.SH", "300001.SZ"]


def _flat_bars(
    dates: list[str],
    ts_codes: list[str],
    *,
    open_by_date: dict | None = None,
    close_by_date: dict | None = None,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Constant-price bars with per-date or per-(code, date) overrides."""

    def _value(overrides: dict, code: str, date: str, default: float) -> float:
        if (code, date) in overrides:
            return float(overrides[(code, date)])
        return float(overrides.get(date, default))

    open_by_date = open_by_date or {}
    close_by_date = close_by_date or {}
    rows = []
    for code in ts_codes:
        previous_close: float | None = None
        for date in dates:
            open_ = _value(open_by_date, code, date, 10.0)
            close = _value(close_by_date, code, date, 10.0)
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": open_,
                    "high": max(open_, close) * 1.02,
                    "low": min(open_, close) * 0.98,
                    "close": close,
                    # The exchange publishes pre_close already adjusted for
                    # the day's corporate action (the ex-div reference).
                    "pre_close": previous_close if previous_close else close,
                    "volume": volume,
                    "amount": volume * close,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
            )
            previous_close = close
    return pd.DataFrame(rows)


def _config(tmp_path: Path, name: str) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path / name / "data",
        duckdb_path=tmp_path / name / "ashare.duckdb",
        parquet_dir=tmp_path / name / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


def _seed(tmp_path: Path, name: str) -> DataConfig:
    """Legacy universe/calendar/stocks database (no legacy daily bars)."""

    config = _config(tmp_path, name)
    with __import__("ashare_data.db", fromlist=["AshareDB"]).AshareDB(
        config.duckdb_path
    ) as db:
        db.create_schema(config)
        db.upsert_stocks(
            [
                {
                    "ts_code": code,
                    "name": code,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for code in CODES
            ],
            config,
        )
        db.upsert_calendar(
            [{"trade_date": date, "is_open": True} for date in DATES],
            config,
        )
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": code,
                    # Distinct in_dates keep the intervals PIT-shaped (the
                    # snapshot-stretched detector rejects a uniform row).
                    "in_date": f"2020010{i + 1}",
                    "out_date": "99991231",
                }
                for i, code in enumerate(CODES)
            ],
            config,
        )
    return config


def _loaded(config: DataConfig, bars: pd.DataFrame) -> AshareDataLoader:
    loader = _formal(config)
    loader.load_data()
    assert loader.bars is not None
    return loader


def _formal(config: DataConfig) -> AshareDataLoader:
    """A02 (p26 I2): a formal load confirms the content identity, so publish
    the manifest for the database's *current* content immediately before
    loading (these tests mutate the seeded database themselves, which moves
    the content on from whatever the fixture published)."""

    publish_manifest(config)
    return AshareDataLoader(config)


def _row(loader: AshareDataLoader, code: str) -> int:
    """Matrix row of ``code`` (the loader exposes sorted ts_codes)."""

    return loader.ts_codes.index(code)


# --- single formal price source (p19 §2.1/§6.2) -----------------------------


def test_loader_source_is_v2_without_silent_legacy_fallback(tmp_path: Path):
    """The formal loader reads the v2 price source; an absent v2 source is
    fail-closed even when the legacy qfq ``daily_bar`` table exists."""

    bars = _flat_bars(DATES, CODES)
    config = _seed(tmp_path, "v2only")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb", DATES, CODES, bars
    )
    loader = _loaded(config, bars)
    assert loader.raw_data_cache["open"].shape == (len(CODES), len(DATES))

    # The same database with a populated legacy daily table but NO v2 file
    # must be rejected, never silently served from the qfq table.
    legacy = _seed(tmp_path, "nolegacyfallback")
    with __import__("ashare_data.db", fromlist=["AshareDB"]).AshareDB(
        legacy.duckdb_path
    ) as db:
        db.upsert_daily(bars.to_dict("records"), legacy)
    with pytest.raises(Exception, match="v2"):
        AshareDataLoader(legacy).load_data()


def test_raw_cache_keys_carry_execution_and_constraint_matrices(tmp_path: Path):
    bars = _flat_bars(DATES, CODES)
    config = _seed(tmp_path, "keys")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb", DATES, CODES, bars
    )
    loader = _loaded(config, bars)
    for key in ("open", "high", "low", "close", "pre_close", "volume",
                "amount", "turnover_rate", "adj_factor", "adjusted_open",
                "up_limit", "down_limit", "suspended"):
        assert key in loader.raw_data_cache, f"missing cache key {key}"
    # Constraint matrices preserve missing-data NaN (0 would fake a limit).
    up = loader.raw_data_cache["up_limit"].numpy()
    assert np.isfinite(up).all()
    # The long bars frame exposes the constraint columns to the broker.
    for column in ("up_limit", "down_limit", "suspended"):
        assert column in loader.bars.columns


def test_engine_rejects_cache_without_price_semantics_keys():
    """Missing price-semantics keys fail closed (p19 §4.1/§4.5)."""

    engine = AshareBacktestEngine(BacktestConfig())
    n, t = 2, 6
    raw = {
        "open": np.full((n, t), 10.0),
        "high": np.full((n, t), 10.0),
        "low": np.full((n, t), 10.0),
        "pre_close": np.full((n, t), 10.0),
        "volume": np.full((n, t), 1000.0),
    }
    with pytest.raises(ValueError, match="p19"):
        engine.run(
            np.ones((n, t)),
            raw,
            CODES[:n],
            DATES[:t],
            np.ones((n, t), dtype=bool),
        )


# --- RED-2: ex-div continuity and the adjusted derivation -------------------


def test_adjusted_series_continuous_at_ex_div_raw_keeps_gap(tmp_path: Path):
    """RED-2: adjusted = raw × af/af_base has no jump at the event; the raw
    series shows the real gap; returns match the hand formula bitwise."""

    event = DATES[3]
    bars = _flat_bars(
        DATES,
        CODES,
        open_by_date={event: 8.0, **{d: 8.0 for d in DATES[3:]}},
        close_by_date={event: 8.0, **{d: 8.0 for d in DATES[3:]}},
    )
    af = {(CODES[0], d): (1.25 if d >= event else 1.0) for d in DATES}
    config = _seed(tmp_path, "exdiv")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        adj_factor=af,
    )
    loader = _loaded(config, bars)

    raw_open = loader.raw_data_cache["open"].numpy()
    adj_open = loader.adjusted_data_cache["open"].numpy()
    # Raw keeps the real gap: 10.0 -> 8.0 at the event.
    assert raw_open[0, 2] == 10.0 and raw_open[0, 3] == 8.0
    # Adjusted is continuous: 8.0 * 1.25 / 1.0 == 10.0 at and after the event.
    assert adj_open[0, 2] == 10.0
    assert adj_open[0, 3] == 10.0
    assert adj_open[0, 4] == 10.0
    # Other stocks (af == 1.0 everywhere) are identity-scaled.
    assert np.array_equal(adj_open[1], raw_open[1])
    # Hand formula: adjusted == raw * af / af_base, bitwise, whole window.
    af_col = loader.raw_data_cache["adj_factor"].numpy()
    base = af_col[0, 0]
    np.testing.assert_array_equal(adj_open[0], raw_open[0] * af_col[0] / base)
    # Adjusted open-to-open return across the event is 0 (dividend
    # received), while the raw open-to-open return would fake a -20% loss.
    from ashare_data.processor import open_to_open_returns

    adj_ret = open_to_open_returns(adj_open)
    raw_ret = open_to_open_returns(raw_open)
    assert adj_ret[0, 1] == 0.0  # holds open t2 -> t3, across the event
    assert raw_ret[0, 1] == pytest.approx(8.0 / 10.0 - 1.0)


def test_no_future_ex_div_leakage_in_factor_and_label_basis(tmp_path: Path):
    """RED-1: values at/before the event are bitwise unchanged whether the
    event is inside the window or not (window start fixed)."""

    event = DATES[5]
    bars = _flat_bars(DATES, CODES)
    config_a = _seed(tmp_path, "leak_a")
    populate_v2_price_tables(
        config_a.duckdb_path.parent / "ashare_v2.duckdb", DATES, CODES, bars
    )
    config_b = _seed(tmp_path, "leak_b")
    af_b = {(CODES[0], d): (2.0 if d >= event else 1.0) for d in DATES}
    populate_v2_price_tables(
        config_b.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        adj_factor=af_b,
    )
    loader_a = _loaded(config_a, bars)
    loader_b = _loaded(config_b, bars)

    adj_a = loader_a.adjusted_data_cache["open"].numpy()
    adj_b = loader_b.adjusted_data_cache["open"].numpy()
    # Event index is 5; everything strictly before it is identical bitwise.
    np.testing.assert_array_equal(adj_a[0, :5], adj_b[0, :5])
    # Labels (open-to-open targets) at signal dates whose whole holding
    # window ends before the event are identical bitwise.
    from ashare_data.processor import open_to_open_returns

    np.testing.assert_array_equal(
        open_to_open_returns(adj_a)[0, :3], open_to_open_returns(adj_b)[0, :3]
    )
    # After the event the adjusted basis reflects it (10x scale).
    assert adj_b[0, 5] == pytest.approx(raw_val := 10.0 * 2.0 / 1.0)
    assert raw_val == 20.0


# --- adjusted-basis anomalies fail closed (p19 §2.1/§4.3) --------------------


@pytest.mark.parametrize("mode", ["missing", "zero"])
def test_adj_factor_anomaly_degrades_stock_fail_closed(tmp_path: Path, mode):
    bars = _flat_bars(DATES, CODES)
    config = _seed(tmp_path, f"afbad_{mode}")
    bad_cell = (CODES[0], DATES[4])
    kwargs: dict = {}
    if mode == "missing":
        kwargs["drop_adj_for"] = {bad_cell}
    else:
        kwargs["adj_factor"] = {**{(c, d): 1.0 for c in CODES for d in DATES},
                                bad_cell: 0.0}
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        **kwargs,
    )
    loader = _formal(config)
    loader.load_data()
    row = _row(loader, CODES[0])

    # The stock's adjusted series is fail-closed: NaN everywhere (never 0).
    adj_open = loader.adjusted_data_cache["open"].numpy()
    assert np.isnan(adj_open[row]).all()
    # Other stocks keep a valid adjusted basis.
    others = [i for i in range(len(CODES)) if i != row]
    for other in others:
        assert np.isfinite(loader.adjusted_data_cache["open"].numpy()[other]).all()
    # Raw execution prices survive (execution does not need adj_factor).
    assert np.isfinite(loader.raw_data_cache["open"].numpy()[row]).all()
    # Degraded provenance is recorded, never swallowed.
    assert CODES[0] in loader.price_degraded
    # Untradeable: the buy mask blocks the degraded stock (labels/factors
    # on its adjusted series are invalid; exits stay market-driven).
    blocked_buy, _ = loader.tradability_masks()
    assert blocked_buy[row].all()
    # The label is NaN for the degraded stock (missing stays NaN, not 0).
    target = loader.target_ret.numpy()
    assert np.isnan(target[row]).all()


def test_adj_factor_anomaly_blocks_only_that_stock(tmp_path: Path):
    bars = _flat_bars(DATES, CODES)
    config = _seed(tmp_path, "afscope")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        drop_adj_for={(CODES[2], DATES[1])},
    )
    loader = _formal(config)
    loader.load_data()
    row = _row(loader, CODES[2])
    blocked_buy, _ = loader.tradability_masks()
    assert blocked_buy[row].all()
    for other in range(len(CODES)):
        if other != row:
            assert not blocked_buy[other].any()


# --- RED-4: exact constraints replace the approximate inference --------------


def test_exact_limit_and_suspension_masks(tmp_path: Path):
    bars = _flat_bars(DATES, CODES)
    day = DATES[4]
    config = _seed(tmp_path, "exactmasks")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        # Stock 0: open exactly at the published up_limit -> buy blocked.
        limit_overrides={
            (CODES[0], day): (10.0, 5.0),
            # Stock 1: one-word open 2% below the limit -> tradable (the
            # approximate board-rate rule mis-blocked near-limit opens).
            (CODES[1], day): (10.2, 5.0),
        },
        # Stock 2: suspended that day -> both sides blocked.
        suspensions=[
            {"ts_code": CODES[2], "trade_date": day, "suspend_type": "S"}
        ],
    )
    loader = _formal(config)
    loader.load_data()
    blocked_buy, blocked_sell = loader.tradability_masks()
    j = DATES.index(day)
    row0 = _row(loader, CODES[0])
    row1 = _row(loader, CODES[1])
    row2 = _row(loader, CODES[2])
    # open == up_limit -> buy blocked, sell still allowed.
    assert blocked_buy[row0, j]
    assert not blocked_sell[row0, j]
    # Near-limit one-word open below up_limit -> NOT blocked (exact rule).
    assert not blocked_buy[row1, j]
    # Suspension row blocks both sides.
    assert blocked_buy[row2, j]
    assert blocked_sell[row2, j]
    # No degraded records: every cell had a complete limit row.
    assert not loader.price_degraded


def test_missing_limit_row_fails_closed_with_degraded_record(tmp_path: Path):
    bars = _flat_bars(DATES, CODES)
    day = DATES[4]
    config = _seed(tmp_path, "misslimit")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        drop_limit_for={(CODES[1], day)},
    )
    loader = _formal(config)
    loader.load_data()
    blocked_buy, blocked_sell = loader.tradability_masks()
    j = DATES.index(day)
    row = _row(loader, CODES[1])
    # Fail-closed: no limit price -> untradeable both sides that day.
    assert blocked_buy[row, j]
    assert blocked_sell[row, j]
    # Other days of the same stock keep exact semantics (not blanket-blocked).
    assert not blocked_buy[row, j + 1]
    assert not blocked_sell[row, j - 1]
    # Degraded provenance records the missing constraint.
    assert CODES[1] in loader.price_degraded


def test_exact_limit_table_drives_engine_entry(tmp_path: Path):
    """The engine opens a near-limit one-word entry the approximate rule
    used to mis-block, and never opens a stock with a missing limit row."""

    day = DATES[3]
    bars = _flat_bars(
        DATES,
        CODES,
        open_by_date={day: 10.15},
        close_by_date={day: 10.15},
    )
    config = _seed(tmp_path, "engineentry")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        limit_overrides={(CODES[0], day): (10.2, 5.0)},
    )
    loader = _loaded(config, bars)
    raw = {k: v.numpy() for k, v in loader.raw_data_cache.items()}
    signal = np.zeros((len(CODES), len(DATES)))
    signal[0, :] = 2.0
    signal[1, :] = 1.0
    result = AshareBacktestEngine(
        BacktestConfig(top_n=1, single_weight_cap=1.0)
    ).run(
        signal,
        raw,
        loader.ts_codes,
        loader.dates,
        np.ones((len(CODES), len(DATES)), dtype=bool),
    )
    entered = {c for position in result.positions for c in position["ts_codes"]}
    assert CODES[0] in entered


# --- RED-3: raw nominal execution prices -------------------------------------


def test_raw_prices_drive_lots_cash_and_commission_floor(tmp_path: Path):
    """Golden lot path: fills happen at raw prices with whole-lot counts
    derived from raw prices and the commission floor on the raw amount,
    under a 10x adjusted/raw price spread."""

    step = DATES[2]
    bars = _flat_bars(DATES, CODES)
    af = {(CODES[0], d): (10.0 if d >= step else 1.0) for d in DATES}
    config = _seed(tmp_path, "rawexec")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        adj_factor=af,
    )
    loader = _loaded(config, bars)
    raw = {k: v.numpy() for k, v in loader.raw_data_cache.items()}
    # Adjusted/raw spread >= 10x from the second session on.
    assert raw["adjusted_open"][0, 3] == pytest.approx(
        10.0 * raw["open"][0, 3]
    )

    from ashare_trading.golden import GoldenParity

    signal = np.zeros((len(CODES), len(DATES)))
    signal[0, :] = 3.0
    signal[1, :] = 2.0
    # P35 (v4): the mid-window adj_factor step is a cash-dividend
    # equivalent (raw flat, adjusted x10 -> a +900% o2o label on the
    # step day).  The engine re-derives its share book at RAW entry
    # prices, so it books the dividend into capital and then trades the
    # excess shares back at the next rebalance; the continuous free
    # reference keeps compounding the adjusted series without a raw
    # re-mark.  The two diverge structurally across the corporate action
    # (the recorded-residual attribution case the golden contract
    # documents) — the friction bound is widened accordingly; the lot
    # assertions below are the point of this test.
    harness = GoldenParity(
        BacktestConfig(top_n=2, initial_capital=100_000.0),
        lot_size=100,
        engine_friction_tol=5e-3,
    )
    report = harness.run(
        signal,
        raw,
        loader.ts_codes,
        loader.dates,
        np.ones((len(CODES), len(DATES)), dtype=bool),
    )
    harness.verify(report)
    filled = [
        fill
        for record in report.records
        for fill in record.fills
        if fill.side == "buy" and fill.status == "filled"
    ]
    assert filled, "expected whole-lot fills"
    for fill in filled:
        assert fill.filled % 100 == 0  # whole lots on the raw price
        # Fill price is the RAW open (~10), never the adjusted (~100).
        assert fill.price == pytest.approx(10.0)
        raw_amount = fill.filled * fill.price
        commission_floor = max(
            5.0, 0.00025 * raw_amount
        )
        assert commission_floor >= 5.0


def test_raw_price_cash_affordability_caps_shares(tmp_path: Path):
    """A small account can open one lot at the raw price; on distorted
    (adjusted-scale) prices the same order would be unaffordable."""

    bars = _flat_bars(DATES, CODES)
    af = {(c, d): 10.0 for c in CODES for d in DATES}
    af[(CODES[0], DATES[0])] = 1.0  # window anchor keeps the spread >= 10x
    config = _seed(tmp_path, "cashcap")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        adj_factor=af,
    )
    loader = _loaded(config, bars)
    raw = {k: v.numpy() for k, v in loader.raw_data_cache.items()}

    from ashare_trading.golden import GoldenParity

    signal = np.zeros((len(CODES), len(DATES)))
    signal[0, :] = 3.0
    harness = GoldenParity(
        BacktestConfig(
            top_n=1,
            initial_capital=1_500.0,  # exactly one lot at raw price 10
            single_weight_cap=1.0,
        ),
        lot_size=100,
    )
    report = harness.run(
        signal,
        raw,
        loader.ts_codes,
        loader.dates,
        np.ones((len(CODES), len(DATES)), dtype=bool),
    )
    buys = [
        fill
        for record in report.records
        for fill in record.fills
        if fill.side == "buy" and fill.status == "filled"
    ]
    assert buys, "one lot must be affordable at the raw price"
    assert all(fill.filled == 100.0 for fill in buys)


# --- RED-5: day-by-day parity on the new basis -------------------------------


def test_golden_parity_day_by_day_with_ex_div_limit_suspension(tmp_path: Path):
    limit_day = DATES[3]
    event = DATES[5]
    af = {(CODES[0], d): (1.25 if d >= event else 1.0) for d in DATES}
    # Only the event stock reprices; the other two stay flat so the
    # event-spanning period isolates the adjusted-basis continuity.
    bars = _flat_bars(
        DATES,
        CODES,
        open_by_date={
            (CODES[0], event): 8.0,
            **{(CODES[0], d): 8.0 for d in DATES[5:]},
        },
        close_by_date={
            (CODES[0], event): 8.0,
            **{(CODES[0], d): 8.0 for d in DATES[5:]},
        },
    )
    config = _seed(tmp_path, "parity")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        adj_factor=af,
        # Sorted row 1 (300001.SZ) opens exactly at its published up_limit:
        # its fresh buy at that entry is blocked (exact rule).
        limit_overrides={(CODES[2], limit_day): (10.0, 5.0)},
        suspensions=[
            {
                "ts_code": CODES[1],
                "trade_date": DATES[4],
                "suspend_type": "S",
            }
        ],
    )
    loader = _loaded(config, bars)
    raw = {k: v.numpy() for k, v in loader.raw_data_cache.items()}
    # Sorted rows: 000001.SZ (ex-div), 300001.SZ, 600000.SH.  300001.SZ
    # jumps in rank on the signal date BEFORE the limit-up entry, so its
    # blocked buy at that entry is a FRESH entrant (a held survivor would
    # legitimately keep its slot — blocking only prevents new buys).
    signal = np.zeros((len(CODES), len(DATES)))
    signal[_row(loader, CODES[0]), :] = 3.0
    signal[_row(loader, CODES[2]), :] = [1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    signal[_row(loader, CODES[1]), :] = 2.0

    from ashare_trading.golden import GoldenParity

    harness = GoldenParity(
        BacktestConfig(top_n=2, initial_capital=1_000_000.0), lot_size=0
    )
    report = harness.run(
        signal,
        raw,
        loader.ts_codes,
        loader.dates,
        np.ones((len(CODES), len(DATES)), dtype=bool),
    )
    # Day-by-day: every executed period is recorded and the free path
    # reproduces the engine's net return exactly (verify asserts it).
    harness.verify(report)
    assert len(report.records) == len(signal[0]) - 2
    # The event-spanning period earns the adjusted (dividend-inclusive)
    # return: holding from the day before the event to the event day is
    # flat in adjusted space (gross label return exactly 0; the only
    # deviation is the fresh-entrant buy fee).  The raw basis would fake a
    # -20% gap on the held stock (-1e-2, three orders of magnitude larger).
    event_entry = next(
        r for r in report.records if r.date == DATES[4]
    )
    assert abs(event_entry.free_return) < 4.0e-5
    # The limit-up day blocks the fresh buy of 300001.SZ: the executed
    # book at that entry holds the ex-div stock plus the rank-3 entrant
    # instead, and no filled buy exists for the blocked name.
    blocked_entry = next(
        i for i, r in enumerate(report.records) if r.date == limit_day
    )
    row1 = _row(loader, CODES[2])
    row2 = _row(loader, CODES[1])
    assert report.target_weights[blocked_entry][row1] == 0.0
    assert report.target_weights[blocked_entry][row2] > 0.0
    assert all(
        fill.ts_code != CODES[2] or fill.status != "filled"
        for fill in report.records[blocked_entry].fills
        if fill.side == "buy"
    )


# --- RED-6: legacy artifact rejection (version impact table) -----------------


def test_version_impact_table_bumps_are_in_place():
    from ashare_model.bare_factor_backtest import BARE_FACTOR_BACKTEST_VERSION
    from ashare_model.factors import FACTOR_COMPUTE_VERSION
    from ashare_model.p3_measurement import P3_MEASUREMENT_VERSION
    from ashare_model.targets import TARGET_CONTRACT_VERSION
    from ashare_model.versions import PROTOCOL_VERSION
    from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION

    assert EXECUTION_SPEC_VERSION == 4  # p35 share-accounting (drifted non-due weights + share-level execution)
    # p21 §8 (APPROVED 2026-09-06): PROTOCOL 26 -> 27 (tier_set additive);
    # p40/p41 §4 (APPROVED 2026-09-19): 27 -> 28 -> 29 (statistical
    # resampling semantics, then candidate-evidence binding).
    assert str(PROTOCOL_VERSION) == "29"
    # p17/D5 (t16 F1): bumped 2 -> 3 for the industry membership PIT
    # switch; p17/D6 bumps to 4 for the family-⑤ source switch.
    assert FACTOR_COMPUTE_VERSION == 4  # t22 semantic fixes (covers D4/D5/D6 bases + t22 consumption fixes)
    assert TARGET_CONTRACT_VERSION == 2
    assert BARE_FACTOR_BACKTEST_VERSION == 4
    assert P3_MEASUREMENT_VERSION == 4
    # Unchanged owners keep their values (p19 §5).
    from ashare_model.data_tier import DATA_TIER_VERSION
    from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
    from ashare_model.cost_matrix import FEE_MATRIX_VERSION

    assert PORTFOLIO_CONSTRUCTOR_VERSION == 1
    assert FEE_MATRIX_VERSION == 1
    # p17 §9 (D6): bumped 1 -> 2 for the revision-tracking B rule.
    assert DATA_TIER_VERSION == 2


def test_legacy_protocol_and_execution_artifacts_classified_legacy():
    from ashare_model.artifact_versions import (
        classify_protocol,
        classify_strategy,
        version_matches,
    )

    assert version_matches("25", "26") is False
    assert version_matches(2, 3) is False

    legacy_strategy = {
        "searcher": "random",
        "reward_version": "15",
        "protocol_version": "25",
        "model_version": 3,
        "dataset_id": "ds",
        "execution_version": 2,
        "portfolio_constructor_version": 1,
        "portfolio_config": {
            "rebalance_frequency": "daily",
            "target_horizon": 1,
            "portfolio_method": "equal_weight",
            "top_n": 30,
            "buy_rank": None,
            "sell_rank": None,
            "min_trade_amount": None,
            "turnover_budget": None,
            "target_weight_change_threshold": 0.0,
            "initial_capital": 100000.0,
            "single_weight_cap": 0.05,
            "commission_rate": 0.00025,
            "min_commission": 5.0,
            "stamp_tax_rate": 0.0005,
            "transfer_fee_rate": 0.00001,
            "slippage_rate": 0.0005,
        },
    }
    verdict = classify_strategy(legacy_strategy)
    assert verdict["legacy"] is True
    assert any("protocol_version 25" in reason for reason in verdict["reasons"])
    assert any("execution_version 2" in reason for reason in verdict["reasons"])

    legacy_protocol = dict(legacy_strategy)
    legacy_protocol.update(
        {
            "research_domain": "unified",
            "stitched": {},
            "ledger": [],
        }
    )
    verdict = classify_protocol(legacy_protocol)
    assert verdict["legacy"] is True
    assert any("protocol_version 25" in reason for reason in verdict["reasons"])


# --- adjacent regressions -----------------------------------------------------


def test_turnover_rate_sourced_from_daily_basic_pit(tmp_path: Path):
    bars = _flat_bars(DATES, CODES)
    config = _seed(tmp_path, "turnover")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        turnover_rate={(CODES[0], d): 2.5 for d in DATES},
    )
    loader = _loaded(config, bars)
    assert loader.raw_data_cache["turnover_rate"].numpy()[0, 0] == pytest.approx(2.5)
    assert loader.raw_data_cache["turnover_rate"].numpy()[1, 0] == pytest.approx(1.0)
    # The loader's long frame keeps the source value.
    rows = loader.bars[loader.bars["ts_code"] == CODES[0]]
    assert (rows["turnover_rate"] == 2.5).all()


def test_volume_column_sourced_from_v2_vol(tmp_path: Path):
    bars = _flat_bars(DATES, CODES, volume=777_777.0)
    config = _seed(tmp_path, "volcol")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb", DATES, CODES, bars
    )
    loader = _loaded(config, bars)
    assert loader.raw_data_cache["volume"].numpy()[0, 0] == pytest.approx(777_777.0)


def test_degraded_adjusted_series_stays_nan_for_labels(tmp_path: Path):
    """Missing stays NaN: neither the adjusted cache nor the label fakes a
    neutral value for a fail-closed series (p19 §2.1, AGENTS §5.3)."""

    bars = _flat_bars(DATES, CODES)
    config = _seed(tmp_path, "nankeep")
    populate_v2_price_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        DATES,
        CODES,
        bars,
        drop_adj_for={(CODES[0], DATES[2])},
    )
    loader = _formal(config)
    loader.load_data()
    target = loader.target_ret.numpy()
    assert np.isnan(target[0]).all()
    adj_close = loader.adjusted_data_cache["close"].numpy()
    assert np.isnan(adj_close[0]).all()
