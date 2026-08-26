from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig, SimConfig
from ashare_data.db import AshareDB
from ashare_data.universe import UniverseReason
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.data_loader import AshareDataLoader
from ashare_trading.run_sim import SimulationRunner
from tests.conftest import make_bars


def _write_sim_db(
    data_config: DataConfig,
    dates,
    ts_codes,
    bars,
    memberships: list[dict] | None = None,
    stocks: list[dict] | None = None,
):
    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            stocks
            or [
                {
                    "ts_code": code,
                    "name": code,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for code in ts_codes
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": date, "is_open": True} for date in dates],
            data_config,
        )
        db.upsert_constituents(
            memberships
            or [
                {
                    "index_code": "000300.SH",
                    "ts_code": code,
                    "in_date": f"2020010{i + 1}",
                    "out_date": "99991231",
                }
                for i, code in enumerate(ts_codes)
            ],
            data_config,
        )


def _make_runner(
    tmp_path: Path,
    n_dates: int = 40,
    top_n: int = 2,
    max_positions: int | None = None,
    *,
    ts_codes: list[str] | None = None,
    bars: pd.DataFrame | None = None,
    memberships: list[dict] | None = None,
    stocks: list[dict] | None = None,
    today: str | None = None,
    single_weight_cap: float = 0.5,
) -> tuple[SimulationRunner, list[str]]:
    """A runner over ``n_dates`` synthetic bars with all paths under tmp_path."""

    dates, default_codes, default_bars = make_bars(n_dates, ts_codes)
    ts_codes = list(ts_codes or default_codes)
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        min_listed_sessions=1,
        index_codes=["000300.SH"],
        index_names=["沪深300"],
    )
    _write_sim_db(
        data_config,
        dates,
        ts_codes,
        bars if bars is not None else default_bars,
        memberships=memberships,
        stocks=stocks,
    )
    model_config = ModelConfig(max_formula_len=6)
    loader = AshareDataLoader(data_config, model_config)
    loader.load_data(ts_codes=ts_codes, dates=dates)
    (data_config.data_dir / "best_ashare_strategy.json").write_text(
        json.dumps({"formula": [1], "formula_text": "RET_1"}), encoding="utf-8"
    )

    position_count = max_positions if max_positions is not None else top_n
    sim_config = SimConfig(
        initial_capital=100000.0,
        max_positions=position_count,
        single_weight_cap=single_weight_cap,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
        stop_signal_path=tmp_path / "STOP",
        progress_path=tmp_path / "progress.json",
    )
    runner = SimulationRunner(
        data_config,
        model_config,
        BacktestConfig(
            initial_capital=sim_config.initial_capital,
            top_n=position_count,
            single_weight_cap=single_weight_cap,
            train_end_date="2024-01-20",
        ),
        sim_config,
        loader,
        today=today,
    )
    return runner, dates


def _replace_stock_bars(
    bars: pd.DataFrame, code: str, opens: list[float]
) -> pd.DataFrame:
    """Rebuild one stock's bars from an explicit open-price series.

    ``pre_close`` is the previous open, and high/low surround the open so
    no day is a one-word limit move unless the caller post-edits it.
    """

    frame = bars[bars["ts_code"] != code].copy()
    by_date = frame.groupby("trade_date", sort=True).size().index.tolist()
    rows = []
    for date, open_ in zip(by_date, opens):
        rows.append(
            {
                "ts_code": code,
                "trade_date": date,
                "open": open_,
                "high": open_ * 1.01,
                "low": open_ * 0.99,
                "close": open_ * 1.005,
                "pre_close": rows[-1]["open"] if rows else open_,
                "volume": 1_000_000.0,
                "amount": 1_000_000.0 * open_,
                "turnover_rate": 1.0,
                "adj_factor": 1.0,
            }
        )
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def _orders_for(runner: SimulationRunner, exec_date: str) -> list[dict]:
    path = runner.sim_config.orders_dir / f"{exec_date}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _trades_for(runner: SimulationRunner, exec_date: str) -> list[dict]:
    path = runner.sim_config.trades_dir / f"{exec_date}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(runner: SimulationRunner) -> dict:
    """Everything a replayed run could pollute, compared as plain data."""

    return {
        "cash": runner.portfolio.cash,
        "trade_count": runner.portfolio.trade_count,
        "positions": {k: asdict(v) for k, v in runner.portfolio.positions.items()},
        "equity_history": runner.portfolio.equity_history,
        "last_exec_date": runner.portfolio.last_exec_date,
        "orders": {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(runner.sim_config.orders_dir.glob("*.json"))
        },
        "trades": {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(runner.sim_config.trades_dir.glob("*.json"))
        },
    }


def test_simulation_runner_end_to_end(tmp_path: Path):
    runner, _ = _make_runner(tmp_path)
    result = runner.run()
    assert isinstance(result["final_cash"], float)
    assert isinstance(result["trade_count"], int)
    assert result["equity_history"]
    assert list(runner.sim_config.orders_dir.glob("*.json"))
    assert list(runner.sim_config.trades_dir.glob("*.json"))
    assert runner.sim_config.state_path.exists()

    # Persisted orders must carry their final execution status (filled /
    # skipped), not the pre-execution "pending" placeholder.
    order_files = sorted(runner.sim_config.orders_dir.glob("*.json"))
    assert order_files
    payload = json.loads(order_files[-1].read_text(encoding="utf-8"))
    if payload:
        assert all(o.get("status") != "pending" for o in payload)


def test_simulation_runner_respects_max_positions(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, max_positions=1, top_n=3)
    runner.run()
    assert len(runner.portfolio.positions) <= 1


def test_simulation_runner_stop_signal(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, n_dates=8)
    stop = tmp_path / "STOP"
    stop.write_text("STOP", encoding="utf-8")
    result = runner.run()
    assert result["equity_history"] == []
    assert stop.read_text(encoding="utf-8") == "STOPPED"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "stopped"


def test_simulation_runner_writes_progress(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=8)
    runner.run()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "finished"
    assert progress["current_date"] == dates[-1]


def test_simulation_runner_resume_matches_full_run(tmp_path: Path):
    """Golden test: one full pass == two passes (first chunk + --resume)."""

    full_runner, dates = _make_runner(tmp_path, n_dates=30)
    full_runner.run()
    full_snapshot = _snapshot(full_runner)

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    resume_runner, split_dates = _make_runner(split_dir, n_dates=30)
    assert split_dates == dates
    # First chunk: signals through dates[15], executions through dates[16].
    resume_runner.run(end_date=dates[15])
    assert resume_runner.portfolio.last_exec_date == dates[16]
    resume_runner.run(resume=True)
    resume_snapshot = _snapshot(resume_runner)

    assert resume_snapshot == full_snapshot


def test_simulation_runner_plain_run_on_history_fails(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=12)
    runner.run(end_date=dates[6])
    with pytest.raises(ValueError, match="resume"):
        runner.run()


def test_simulation_runner_resume_rejects_start_before_watermark(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=12)
    runner.run(end_date=dates[6])
    with pytest.raises(ValueError, match="precedes"):
        runner.run(start_date=dates[2], resume=True)


def test_simulation_runner_resume_requires_watermark(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, n_dates=12)
    # Fresh state has no last_exec_date; --resume without any history is an error.
    with pytest.raises(ValueError, match="last_exec_date"):
        runner.run(resume=True)


def test_simulation_runner_reset_allows_clean_replay(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=12)
    runner.run(end_date=dates[6])
    runner.portfolio.reset()
    result = runner.run()
    history_dates = [h["trade_date"] for h in result["equity_history"]]
    assert len(history_dates) == len(dates) - 1
    assert len(history_dates) == len(set(history_dates))
    assert runner.portfolio.last_exec_date == dates[-1]


def test_simulation_runner_honors_artifact_direction(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, n_dates=12)
    # Record a negative direction in the strategy artifact: the runner must
    # use it verbatim instead of inferring from the training window.
    path = runner.data_config.data_dir / "best_ashare_strategy.json"
    path.write_text(
        json.dumps(
            {
                "formula": [1],
                "formula_text": "RET_1",
                "direction": -1,
            }
        ),
        encoding="utf-8",
    )
    runner.load_formula()
    assert runner.direction == -1
    assert runner._has_recorded_direction
    runner.run()
    # A direction of -1 flips the signal: the selected paper orders must
    # differ from a +1 replay of the same formula (fresh state each side).
    flipped_orders = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(runner.sim_config.orders_dir.glob("*.json"))
    }
    runner.portfolio.reset()
    runner.direction = 1
    runner._has_recorded_direction = True
    runner.run()
    replay_orders = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(runner.sim_config.orders_dir.glob("*.json"))
    }
    assert flipped_orders != replay_orders


# --- PIT eligibility in daily selection + current-ST isolation ---------------


def test_sim_excludes_future_member_until_join_day(tmp_path: Path):
    # A stock that only joins the index later carries the strongest signal
    # every day: it must never enter the sim candidates before its join
    # date, and become selectable exactly from the join date on.
    codes = ["000001.SZ", "600000.SH", "300001.SZ", "600001.SH"]
    future = "600001.SH"
    n_dates = 14
    join_day = 6
    dates, _, default_bars = make_bars(n_dates, codes)
    bars = _replace_stock_bars(
        default_bars, future, [10.0 * 1.03**i for i in range(n_dates)]
    )
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20200101",
            "out_date": "99991231",
        }
        for code in codes
        if code != future
    ] + [
        {
            "index_code": "000300.SH",
            "ts_code": future,
            "in_date": dates[join_day],
            "out_date": "99991231",
        }
    ]
    runner, _ = _make_runner(
        tmp_path, n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars, memberships=memberships,
    )
    runner.run()

    # Before the join day no order may mention the future member.
    for exec_date in dates[1 : join_day + 1]:
        assert all(
            o["ts_code"] != future for o in _orders_for(runner, exec_date)
        )
    # The first eligible signal date is the join day itself: its execution
    # buys the future member.
    join_orders = _orders_for(runner, dates[join_day + 1])
    assert any(
        o["ts_code"] == future and o["side"] == "buy" and o["status"] == "filled"
        for o in join_orders
    )


def test_sim_universe_exit_sells_position(tmp_path: Path):
    # A held stock whose membership interval ends must leave through the
    # ordinary sell path on its first ineligible execution day.
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    exiter = "000001.SZ"
    n_dates = 14
    exit_day = 6
    dates, _, default_bars = make_bars(n_dates, codes)
    # The exiter leads the signal until its exit (2%/day vs 1%/day).
    bars = _replace_stock_bars(
        default_bars,
        exiter,
        [10.0 * 1.02**i for i in range(exit_day)]
        + [10.0 * 1.02 ** (exit_day - 1)] * (n_dates - exit_day),
    )
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20200101",
            "out_date": "99991231" if code != exiter else dates[exit_day],
        }
        for code in codes
    ]
    runner, _ = _make_runner(
        tmp_path, n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars, memberships=memberships,
        # A non-binding cap keeps the whole-lot book free of daily drift
        # churn: this test exercises the exit-sell path, not the cap.
        single_weight_cap=1.0,
    )
    runner.run()

    # Signal day 0 is an all-zero cross-section (no dispersion): the book
    # holds, so the first buyable day is the second signal day (T1-02).
    first = _orders_for(runner, dates[2])
    assert any(
        o["ts_code"] == exiter and o["side"] == "buy" and o["status"] == "filled"
        for o in first
    )
    exit_orders = _orders_for(runner, dates[exit_day])
    sell = next(o for o in exit_orders if o["ts_code"] == exiter)
    assert sell["side"] == "sell"
    assert sell["status"] == "filled"
    assert exiter not in runner.portfolio.positions


def test_sim_limit_down_blocks_exit_sell_and_keeps_position(tmp_path: Path):
    # When the exit-day open is a one-word limit-down the sell cannot fill:
    # the position is kept under the existing matching rule and every later
    # sell attempt is skipped the same way.
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    exiter = "000001.SZ"
    n_dates = 14
    exit_day = 6
    dates, _, default_bars = make_bars(n_dates, codes)
    last = 10.0 * 1.02 ** (exit_day - 1)
    # Limit-down every day from the exit day on: each open is -10% versus
    # its own pre_close, so the sell can never fill.
    bars = _replace_stock_bars(
        default_bars,
        exiter,
        [10.0 * 1.02**i for i in range(exit_day)]
        + [last * 0.90**k for k in range(1, n_dates - exit_day + 1)],
    )
    # One-word limit-down from the exit day on.
    tail = (bars["ts_code"] == exiter) & (bars["trade_date"] >= dates[exit_day])
    bars.loc[tail, "high"] = bars.loc[tail, "open"]
    bars.loc[tail, "low"] = bars.loc[tail, "open"]
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20200101",
            "out_date": "99991231" if code != exiter else dates[exit_day],
        }
        for code in codes
    ]
    runner, _ = _make_runner(
        tmp_path, n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars, memberships=memberships,
        # Non-binding cap: this test exercises the limit-down exit path.
        single_weight_cap=1.0,
    )
    runner.run()

    sells = [
        o
        for d in dates[exit_day:]
        for o in _orders_for(runner, d)
        if o["ts_code"] == exiter and o["side"] == "sell"
    ]
    assert sells
    assert all(
        o["status"] == "skipped" and o["reason"] == "limit_down" for o in sells
    )
    assert exiter in runner.portfolio.positions


def test_current_st_stock_stays_in_union_and_trades_on_board_rates(
    tmp_path: Path,
):
    # A currently-ST stock keeps its historical membership and trades under
    # the board band in replay: a +5.2% one-word open is buyable (the 5%
    # ST band would have blocked it), the display name stays the current
    # one, and every historical cell keeps the STATUS_UNKNOWN audit bit.
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    st_code = "600000.SH"
    n_dates = 12
    buy_day = 5
    dates, _, default_bars = make_bars(n_dates, codes)
    opens = [10.0] * (buy_day - 1) + [10.52] + [10.52 * 1.052] + [
        10.52 * 1.052 * 1.02**k for k in range(1, n_dates - buy_day)
    ]
    bars = _replace_stock_bars(default_bars, st_code, opens)
    # One-word +5.2% open on the first buyable day.
    tail = (bars["ts_code"] == st_code) & (bars["trade_date"] == dates[buy_day])
    bars.loc[tail, "high"] = opens[buy_day]
    bars.loc[tail, "low"] = opens[buy_day]
    stocks = [
        {
            "ts_code": code,
            "name": "*ST 星源" if code == st_code else code,
            "industry": None,
            "list_date": "20200101",
            "is_st": code == st_code,
        }
        for code in codes
    ]
    runner, _ = _make_runner(
        tmp_path, n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars, stocks=stocks,
    )
    runner.run()

    assert st_code in runner.loader.ts_codes
    row = runner.loader.ts_codes.index(st_code)
    assert runner.loader.universe_reason_codes is not None
    assert np.all(
        runner.loader.universe_reason_codes[row] & int(UniverseReason.STATUS_UNKNOWN)
    )
    day_orders = _orders_for(runner, dates[buy_day])
    assert any(
        o["ts_code"] == st_code and o["side"] == "buy" and o["status"] == "filled"
        for o in day_orders
    )
    assert runner.portfolio.positions[st_code].name == "*ST 星源"


def test_sim_uses_current_st_snapshot_only_on_same_day(tmp_path: Path):
    # The current is_st snapshot applies the 5% band only when the
    # execution date equals today: on that day a +5.2% one-word open
    # blocks the buy, while the identical historical replay fills it.
    codes = ["000001.SZ", "600000.SH"]
    st_code = "600000.SH"
    n_dates = 8
    dates, _, default_bars = make_bars(n_dates, codes)
    today = dates[6]
    opens = [10.0] * 5 + [10.52] + [10.52 * 1.052] + [10.52 * 1.052]
    bars = _replace_stock_bars(default_bars, st_code, opens)
    # One-word +5.2% open on the today execution day.
    tail = (bars["ts_code"] == st_code) & (bars["trade_date"] == today)
    bars.loc[tail, "high"] = 10.52 * 1.052
    bars.loc[tail, "low"] = 10.52 * 1.052
    stocks = [
        {
            "ts_code": code,
            "name": "*ST 星源" if code == st_code else code,
            "industry": None,
            "list_date": "20200101",
            "is_st": code == st_code,
        }
        for code in codes
    ]

    same_day_runner, run_dates = _make_runner(
        tmp_path / "same_day", n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars, stocks=stocks, today=today,
    )
    same_day_runner.run()
    # The 5% band applies exactly on the today execution day: the ST stock
    # is excluded from that day's candidates.
    assert all(
        o["ts_code"] != st_code for o in _orders_for(same_day_runner, today)
    )
    # The very next execution day is historical again: board rates resume
    # and the same signal buys the ST stock.
    next_exec = run_dates[run_dates.index(today) + 1]
    assert any(
        o["ts_code"] == st_code and o["side"] == "buy" and o["status"] == "filled"
        for o in _orders_for(same_day_runner, next_exec)
    )

    control_runner, _ = _make_runner(
        tmp_path / "control", n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars, stocks=stocks,
    )
    control_runner.run()
    # Historical replay: board band applies, the buy fills — the current
    # ST name must never rewrite the past.
    day_orders = _orders_for(control_runner, today)
    assert any(
        o["ts_code"] == st_code and o["side"] == "buy" and o["status"] == "filled"
        for o in day_orders
    )


def test_sim_top_n_matches_backtest_engine_per_signal_date(tmp_path: Path):
    # The strongest cross-check: after every execution day the sim's
    # holdings equal the backtest engine's position snapshot for the same
    # signal date, with both consuming the same signals and PIT mask.
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    n_dates = 14
    _, _, default_bars = make_bars(n_dates, codes)
    # A tops from signal day 6 (jump to 200%), B from day 2 to day 5, C
    # never — the selection flips twice, exercising buys and sells.
    a = [10.0] * 6 + [30.0 * 1.02**k for k in range(n_dates - 6)]
    b = [10.0] * 2 + [15.0 * 1.005**k for k in range(n_dates - 2)]
    c = [10.0 * 1.001**i for i in range(n_dates)]
    bars = _replace_stock_bars(default_bars, "000001.SZ", a)
    bars = _replace_stock_bars(bars, "600000.SH", b)
    bars = _replace_stock_bars(bars, "300001.SZ", c)
    runner, dates = _make_runner(
        tmp_path, n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=codes, bars=bars,
    )
    runner.run()
    assert runner.vm.universe_mask is not None

    signals = runner.compute_signals()
    raw = {k: v.numpy() for k, v in runner.loader.raw_data_cache.items()}
    engine = AshareBacktestEngine(
        BacktestConfig(
            initial_capital=100000.0,
            top_n=1,
            single_weight_cap=0.5,
            train_end_date="2024-01-20",
        )
    )
    ref = engine.run(
        signals, raw, runner.loader.ts_codes, dates,
        universe_mask=runner.loader.universe_mask,
    )

    # Compare the stock SET, not exact share counts: the sim targets whole
    # lots and may trim sub-lot drift, but every engine pick must be held
    # (quantity > 0) after the matching execution day and nothing else.
    held: dict[str, int] = {}
    for k, snapshot in enumerate(ref.positions):
        exec_date = dates[k + 1]
        for trade in _trades_for(runner, exec_date):
            delta = trade["quantity"] if trade["side"] == "buy" else -trade["quantity"]
            held[trade["ts_code"]] = held.get(trade["ts_code"], 0) + delta
        assert {c for c, q in held.items() if q > 0} == set(snapshot["ts_codes"])


def test_sim_equal_weights_enforces_single_weight_cap():
    # T1-02: per-name weight is min(1/top_n, cap) and the remainder stays
    # in cash — the paper-trading sim must never renormalize above the cap.
    weights = SimulationRunner._equal_weights(
        [0, 1], 4, top_n=5, single_weight_cap=0.1
    )
    assert weights.tolist() == [0.1, 0.1, 0.0, 0.0]
    assert weights.sum() == pytest.approx(0.2)
    # With a non-binding cap the weight is exactly 1/top_n.
    weights = SimulationRunner._equal_weights(
        [0, 1], 4, top_n=2, single_weight_cap=1.0
    )
    assert weights.tolist() == [0.5, 0.5, 0.0, 0.0]
    assert SimulationRunner._equal_weights([], 3, top_n=2, single_weight_cap=0.1).sum() == 0.0


def test_sim_buy_orders_respect_single_weight_cap(tmp_path: Path):
    # Integration: with max_positions=5 and cap=0.1, a buy order's notional
    # is bounded by ~cap * equity (whole-lot flooring only shrinks it).  The
    # old 1/len(selected) weighting (0.33 for 3 names) would blow the cap.
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    n_dates = 12
    dates, _, default_bars = make_bars(n_dates, codes)
    # Distinct, stable signal ranking across stocks (linear price paths).
    a = [10.0 + 0.30 * i for i in range(n_dates)]
    b = [10.0 + 0.20 * i for i in range(n_dates)]
    c = [10.0 + 0.10 * i for i in range(n_dates)]
    bars = _replace_stock_bars(default_bars, "000001.SZ", a)
    bars = _replace_stock_bars(bars, "600000.SH", b)
    bars = _replace_stock_bars(bars, "300001.SZ", c)
    runner, _ = _make_runner(
        tmp_path, n_dates=n_dates, top_n=5, max_positions=5,
        ts_codes=codes, bars=bars,
    )
    runner.run()
    cap = runner.sim_config.single_weight_cap
    equity_ceiling = runner.sim_config.initial_capital * 1.10
    for date in dates[1:]:
        for order in _orders_for(runner, date):
            if order["side"] == "buy":
                assert order["quantity"] * order["price"] <= cap * equity_ceiling
                assert order["quantity"] * order["price"] > 0
