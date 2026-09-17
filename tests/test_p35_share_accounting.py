"""P35 share-accounting semantic repair (executable projection).

Authority chain: ``docs/p35_share_accounting_draft.md`` (APPROVED via user
direct instruction 2026-09-17, work package 1).  Every expectation below
is an INDEPENDENT hand computation over synthetic deterministic prices
(raw == adjusted unless a split is under test) — never derived from the
implementation.  The five contract invariants:

* R-A round-trip: prices that return to their entry levels must return
  the NAV to exactly 1.0 under zero fees (the frozen-weight engine
  harvests a phantom 10100/9801 from its implicit daily re-normalization
  and fails);
* R-B no-trade drift: with no executions the share counts are constant
  and the weights drift with prices (hand: 0.55, 0.59900990…, final
  equity exactly 1.03);
* R-C orders come from the DRIFTED holdings (hand: sell 1,200 shares of
  A / buy 2,400 of B after A doubled — the frozen engine trades nothing);
* R-D reconciliation: cash + Σ shares×price == equity − fees at the
  execution instant, fees from the cost model on the achieved notionals,
  buys lot-floored and cash-capped (hand: 1,499,290 = 1,500,000 − 710);
* R-E boundaries: a sell-blocked name keeps its drifted weight and gets
  zero orders (the cash boundary stops the unfunded buy), and a 2:1
  split is drift-neutral with the share book re-derived at the next
  execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_model.backtest import AshareBacktestEngine

LOT = 100


def _bt_config(**over) -> BacktestConfig:
    fields = dict(
        portfolio_method="equal_weight",
        buy_rank=2,
        sell_rank=2,
        top_n=2,
        single_weight_cap=0.5,
        initial_capital=1_000_000.0,
        min_trade_amount=None,
        target_weight_change_threshold=0.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    fields.update(over)
    return BacktestConfig(**fields)


def _raw_cache(open_: np.ndarray, adjusted_open: np.ndarray | None = None,
               up_limit: np.ndarray | None = None,
               down_limit: np.ndarray | None = None,
               suspended: np.ndarray | None = None) -> dict:
    wide_open = (up_limit if up_limit is not None
                 else open_ * 10.0)
    deep_open = (down_limit if down_limit is not None
                 else open_ * 0.1)
    susp = (suspended if suspended is not None
            else np.zeros_like(open_))
    return {
        "open": open_,
        "volume": np.ones_like(open_),
        "adjusted_open": (adjusted_open if adjusted_open is not None
                          else open_.copy()),
        "up_limit": wide_open,
        "down_limit": deep_open,
        "suspended": susp,
    }


def _run(open_: np.ndarray, rebalance_mask: np.ndarray,
         adjusted_open: np.ndarray | None = None,
         down_limit: np.ndarray | None = None, **config_over) -> object:
    n_stocks, n_dates = open_.shape
    engine = AshareBacktestEngine(_bt_config(**config_over))
    # A constant, dispersive signal: stock 0 always ranks ahead of stock 1,
    # so every due rebalance targets the equal-weight book of both names.
    factors = np.vstack([
        np.full(n_dates, 1.0),
        np.full(n_dates, 0.5),
    ])
    return engine.run(
        factors,
        _raw_cache(open_, adjusted_open, down_limit=down_limit),
        [f"{i:06d}.SZ" for i in range(n_stocks)],
        [f"202601{d:02d}" for d in range(1, n_dates + 1)],
        universe_mask=np.ones_like(open_, dtype=bool),
        rebalance_mask=rebalance_mask,
    )


class TestRoundTripToPar:
    def test_round_trip_returns_nav_to_one(self):
        # Funding executes at open1 (A=110, B=45); the prices return to
        # those entry levels at open3 and open4 is flat:
        # A: 100 -> [110 -> 90 -> 110] ; B: 50 -> [45 -> 55 -> 45].
        # A share-constant book must end at exactly 1.0 (the drifted
        # weights wander and return; the frozen-weight engine harvests a
        # phantom (101/99)^2 - 1 from its implicit daily re-normalization).
        open_ = np.array([
            [100.0, 110.0, 90.0, 110.0, 110.0],
            [50.0, 45.0, 55.0, 45.0, 45.0],
        ])
        mask = np.array([True, False, True, False, False])
        result = _run(open_, mask)
        assert result.equity_curve[-1] == pytest.approx(1.0, abs=1e-9)


class TestNoTradeDrift:
    def test_weights_drift_shares_constant(self):
        # Funding at open1 (both 100); A compounds +10% per session and
        # B -10%, never rebalanced.  Hand path (share view):
        #   open2: A 550k, B 450k, E = 1,000,000, w_A = 0.55
        #   open3: A 605k, B 405k, E = 1,010,000, w_A = 0.605/1.01
        #   open4: A 665.5k, B 364.5k, E = 1,030,000, w_A = 0.6655/1.03
        open_ = np.array([
            [100.0, 100.0, 110.0, 121.0, 133.1],
            [100.0, 100.0, 90.0, 81.0, 72.9],
        ])
        mask = np.array([True, False, False, False, False])
        result = _run(open_, mask)
        assert result.equity_curve[-1] == pytest.approx(1.03, abs=1e-9)
        # positions[t] marks the book at that session's ENTRY open: the
        # last iterated session marks open3 (drifted 0.605/1.01, rounded
        # to 6 decimals in the record).
        last = result.positions[-1]["weights"]
        assert last[0] == pytest.approx(0.605 / 1.01, abs=1e-6)
        assert last[1] == pytest.approx(0.405 / 1.01, abs=1e-6)
        # shares constant across the no-trade span
        assert result.final_shares[0] == pytest.approx(5_000.0, abs=1e-6)
        assert result.final_shares[1] == pytest.approx(5_000.0, abs=1e-6)
        assert result.final_cash == pytest.approx(0.0, abs=1e-6)


class TestOrdersFromDriftedHoldings:
    def test_rebalance_trades_the_drift_not_the_frozen_target(self):
        # A doubles while B is flat; the t1 rebalance (entry open2) must
        # sell A down and buy B up measured from the DRIFTED book
        # (w = 2/3, 1/3 on 1.5M), not from the frozen 50/50.
        open_ = np.array([
            [100.0, 100.0, 200.0, 200.0, 200.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ])
        mask = np.array([True, True, False, False, False])
        result = _run(open_, mask)
        # Hand: sells 1,200 A @200 = 240,000 (lot floor from 1,250);
        # buys 2,400 B @100 = 240,000 (cash-capped from 2,500); zero fees
        # -> the full proceeds are spent and the cash residual is zero.
        assert result.sell_weights[1][0] == pytest.approx(240_000 / 1_500_000)
        assert result.buy_weights[1][1] == pytest.approx(240_000 / 1_500_000)
        assert result.final_shares[0] == pytest.approx(3_800.0, abs=1e-6)
        assert result.final_shares[1] == pytest.approx(7_400.0, abs=1e-6)
        assert result.final_cash == pytest.approx(0.0, abs=1e-6)
        # achieved book weights at the execution (share view:
        # A 3,800x200 = 760,000; B 7,400x100 = 740,000; cash 0)
        weights = result.positions[1]["weights"]
        assert weights[0] == pytest.approx(760_000 / 1_500_000, abs=1e-6)
        assert weights[1] == pytest.approx(740_000 / 1_500_000, abs=1e-6)


class TestReconciliation:
    def test_cash_shares_equity_fees_reconcile(self):
        # Same doubling scenario with real fees: commission 10bp both
        # sides, stamp 10bp sell.  Hand chain (fees hit the funding too):
        #   t0 funding: 5,000 shares each = 1,000,000 + 1,000 fees > cash
        #     -> lot-reduced to 4,900 each (980,000), fees 980,
        #     cash 19,020, E = 999,020
        #   t0 return: A doubles, B flat -> E(open2) = 1,489,020
        #     (share view: 4,900x200 + 4,900x100 + 19,020)
        #   t1 execution (entry open A=200, B=100):
        #     sells 1,100 A @200 = 220,000 (lot floor from 1,177.45)
        #       -> sell fees 220 + 220 = 440; cash = 238,580
        #     buys B: 2,500 unaffordable -> 2,400 unaffordable
        #       -> 2,300 @100 = 230,000, fee 230; cash = 8,350
        #     E_after = 8,350 + 3,800x200 + 7,200x100 = 1,488,350
        #     (= 1,489,020 - 670 total fees, exact)
        open_ = np.array([
            [100.0, 100.0, 200.0, 200.0, 200.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ])
        mask = np.array([True, True, False, False, False])
        result = _run(
            open_, mask,
            commission_rate=0.001,
            stamp_tax_rate=0.001,
        )
        assert result.final_shares[0] == pytest.approx(3_800.0, abs=1e-6)
        assert result.final_shares[1] == pytest.approx(7_200.0, abs=1e-6)
        assert result.final_cash == pytest.approx(8_350.0, abs=1e-6)
        # the reconciliation identity at the execution instant
        assert result.final_cash + result.final_shares[0] * 200.0 + (
            result.final_shares[1] * 100.0
        ) == pytest.approx(1_488_350.0, abs=1e-6)
        assert result.equity_curve[-1] == pytest.approx(
            1_488_350.0 / 1_000_000.0, abs=1e-6
        )


class TestBoundaries:
    def test_sell_blocked_keeps_drifted_weight_and_stops_unfunded_buy(self):
        # A: 100 -> 100 -> 110 ; B: 100 -> 100 -> 150.  Funded 50/50 at
        # open1 (both 100); drifted at open2 (share view: A 500k, B 750k
        # on 1.25M): w = (0.4, 0.6).  The t1 rebalance targets 50/50, but
        # B's sell is blocked at the entry open (open == down_limit), so
        # B stays at its drifted weight and A's unfunded buy is capped to
        # zero by the cash boundary: NO trades at all.
        open_ = np.array([
            [100.0, 100.0, 110.0, 110.0, 110.0],
            [100.0, 100.0, 150.0, 150.0, 150.0],
        ])
        down_limit = np.full_like(open_, 1.0)
        down_limit[1, 2] = 150.0  # B sell-blocked on the entry day
        mask = np.array([True, True, False, False, False])
        result = _run(open_, mask, down_limit=down_limit)
        weights = result.positions[1]["weights"]
        assert weights[0] == pytest.approx(0.55 / 1.30, abs=1e-6)
        assert weights[1] == pytest.approx(0.75 / 1.30, abs=1e-6)
        assert result.sell_weights[1][1] == 0.0
        assert result.buy_weights[1][0] == 0.0

    def test_split_is_drift_neutral_and_shares_rederive(self):
        # A splits 2:1 between open1 and open2: raw open halves (100 ->
        # 50) while the adjusted series stays 100 — the adjusted o2o is
        # drift-neutral, and the next execution re-derives doubled shares
        # from the halved raw price.
        open_ = np.array([
            [100.0, 100.0, 50.0, 50.0, 50.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ])
        adjusted = np.array([
            [100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ])
        mask = np.array([True, False, True, False, False])
        result = _run(open_, mask, adjusted_open=adjusted)
        # drift neutrality: the book weights never move before the t2
        # execution (both legs flat on the adjusted basis)
        weights = result.positions[1]["weights"]
        assert weights[0] == pytest.approx(0.5, abs=1e-6)
        assert weights[1] == pytest.approx(0.5, abs=1e-6)
        # the t2 execution re-derives shares at the halved raw open:
        # 0.5 * 1,000,000 / 50 = 10,000 shares (double the funded 5,000)
        assert result.final_shares[0] == pytest.approx(10_000.0, abs=1e-6)
        assert result.final_shares[1] == pytest.approx(5_000.0, abs=1e-6)
        assert result.equity_curve[-1] == pytest.approx(1.0, abs=1e-6)
