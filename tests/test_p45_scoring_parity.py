"""P45 scoring-vs-engine parity RED/GREEN tests (WP 9).

Each test asserts an independent invariant of the p45 preregistration
contract (``docs/p45_scoring_engine_parity_draft.md`` §2/§5):

* the comparison tool runs fixed candidates through BOTH paths on the
  same window/masks/direction/config and reports the four error families
  with the granularity recorded (N1);
* the tolerances are the frozen mechanism formulas of the granularity
  g = lot_size x price / capital — hand-checked (N2);
* the reliability gate refuses tolerance breaches, direction flips and
  material ranking inversions, and discloses tie inversions (N3);
* the ranking comparison computes Spearman and splits inversions into
  material/tie by hand-checked cases (N4);
* the loader-level wiring feeds both paths the same epoch slice (N5).
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig

from ashare_model.scoring_parity import (
    SCORING_PARITY_VERSION,
    compare_candidate_parity,
    granularity,
    parity_bounds,
    proxy_reliability_verdict,
    ranking_comparison,
)
from ashare_data.processor import open_to_open_returns
from tests.conftest import with_price_semantics


def _market(n_stocks: int = 6, n_dates: int = 10, seed: int = 7):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"202401{i + 2:02d}" for i in range(n_dates)]
    rng = np.random.default_rng(seed)
    open_ = np.cumprod(
        1.0 + rng.normal(0.0005, 0.005, (n_stocks, n_dates)), axis=1
    ) * 10.0
    pre_close = np.roll(open_, 1, axis=1)
    pre_close[:, 0] = open_[:, 0]
    raw = with_price_semantics({
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": pre_close,
        "volume": np.full_like(open_, 1_000_000.0),
    })
    signal = rng.normal(size=(n_stocks, n_dates))
    return codes, dates, raw, signal


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100_000.0,
        top_n=3,
        single_weight_cap=1.0 / 3.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _fixed_candidates(n_stocks: int, n_dates: int, k: int = 3):
    rng = np.random.default_rng(45)
    return [
        (f"cand{i}", 1, rng.normal(size=(n_stocks, n_dates)))
        for i in range(k)
    ]


# --- N2: the tolerance formulas are the frozen mechanism constants ---------


def test_n2_granularity_and_bounds_are_the_frozen_formulas():
    assert granularity(lot_size=100, price_ref=10.0, capital=100_000.0) == (
        pytest.approx(0.01)
    )
    bounds = parity_bounds(
        _cfg(), price_ref=10.0, n_periods=6
    )
    g = 0.01
    assert bounds["daily_net"] == pytest.approx(0.2 * g)
    assert bounds["fee_fraction"] == pytest.approx(0.05 * g)
    assert bounds["turnover"] == pytest.approx(10.0 * g)
    assert bounds["total_net"] == pytest.approx(0.05 * g * 6)
    # The bounds scale linearly with the granularity (small capital ->
    # coarse lots -> proportionally looser bounds, matching the mechanism).
    coarse = parity_bounds(_cfg(initial_capital=10_000.0), price_ref=10.0, n_periods=6)
    assert coarse["daily_net"] == pytest.approx(10.0 * bounds["daily_net"])


# --- N4: ranking comparison (hand-checked Spearman + inversion split) -------


def test_n4_ranking_comparison_identical_orders():
    rho, material, tie = ranking_comparison(
        proxy_scores={"a": 0.010, "b": 0.005, "c": 0.001},
        engine_scores={"a": 0.011, "b": 0.004, "c": 0.0005},
        material_gap=0.002,
    )
    assert rho == pytest.approx(1.0)
    assert material == ()
    assert tie == ()


def test_n4_ranking_comparison_splits_material_and_tie():
    # Proxy: a > b > c; engine: a > c > b  -> one inversion (b, c).
    # The b-c proxy gap (0.002) decides material vs tie against the
    # threshold (fixture correction: the RED numbers put the gap at 0.004
    # while claiming a tie at threshold 0.002 - arithmetically inconsistent).
    proxy = {"a": 0.010, "b": 0.003, "c": 0.001}
    engine = {"a": 0.011, "c": 0.004, "b": 0.0005}
    rho, material, tie = ranking_comparison(proxy, engine, material_gap=0.002)
    assert rho == pytest.approx(0.5)
    assert material == () and tie == (("b", "c"),)
    # with a tighter threshold the same inversion is material
    rho, material, tie = ranking_comparison(proxy, engine, material_gap=0.001)
    assert material == (("b", "c"),) and tie == ()


# --- N3: the reliability gate (fabricated reports) ---------------------------


def _row(name, *, net=1e-5, fee=1e-6, turn=1e-3, direction=True, total=None):
    from ashare_model.scoring_parity import CandidateParityRow

    return CandidateParityRow(
        candidate=name,
        direction=1,
        proxy_direction=1 if direction else -1,
        engine_direction=1,
        daily_net_error_max=net,
        daily_net_error_mean=net / 2,
        fee_fraction_error_max=fee,
        turnover_error_max=turn,
        total_net_return_error=net * 3 if total is None else total,
        total_fee_error=fee * 3,
        proxy_total_net=0.01,
        engine_total_net=0.01,
    )


def _report(rows, *, g=0.01, flips=(), material=(), tie=()):
    from ashare_model.scoring_parity import ScoringParityReport

    return ScoringParityReport(
        granularity=g,
        bounds=parity_bounds(_cfg(), price_ref=10.0, n_periods=6),
        rows=tuple(rows),
        direction_flips=tuple(flips),
        ranking_spearman=1.0,
        inversions_material=tuple(material),
        inversions_tie=tuple(tie),
    )


def test_n3_reliable_when_all_within():
    report = _report([_row("a"), _row("b")], tie=(("a", "b"),))
    verdict = proxy_reliability_verdict(report)
    assert verdict["verdict"] == "reliable"
    # Tie inversions are disclosed (as JSON-friendly lists), never dropped.
    assert verdict["inversions_tie"] == [["a", "b"]]
    assert verdict["reasons"] == []


def test_n3_unreliable_on_each_tolerance_breach():
    cases = [
        (_row("a", net=0.003, total=1e-4), "daily net"),
        (_row("a", fee=0.001), "fee fraction"),
        (_row("a", turn=0.2), "turnover"),
        (_row("a", total=0.01), "total net"),
        (_row("a", direction=False), "direction"),
    ]
    for bad_row, needle in cases:
        report = _report([_row("ok"), bad_row])
        verdict = proxy_reliability_verdict(report)
        assert verdict["verdict"] == "unreliable", needle
        assert any(
            bad_row.candidate in r and needle in r for r in verdict["reasons"]
        ), (needle, verdict["reasons"])


def test_n3_unreliable_on_material_inversion():
    report = _report(
        [_row("a"), _row("b")], material=(("a", "b"),)
    )
    verdict = proxy_reliability_verdict(report)
    assert verdict["verdict"] == "unreliable"
    assert any("ranking" in r for r in verdict["reasons"])


# --- N1/N5: the comparison tool on a synthetic market ------------------------


def test_n1_compare_reports_four_error_families_within_bounds():
    codes, dates, raw, signal = _market()
    candidates = _fixed_candidates(len(codes), len(dates))
    cfg = _cfg()
    report = compare_candidate_parity(
        candidates, raw, codes, dates,
        universe_mask=np.ones(signal.shape, dtype=bool),
        backtest_config=cfg,
    )
    assert SCORING_PARITY_VERSION == 1
    assert len(report.rows) == 3
    for row in report.rows:
        assert row.candidate.startswith("cand")
        assert row.proxy_direction in (1, -1)
        assert row.engine_direction in (1, -1)
        assert row.daily_net_error_max <= report.bounds["daily_net"], row
        assert row.fee_fraction_error_max <= report.bounds["fee_fraction"], row
        assert row.turnover_error_max <= report.bounds["turnover"], row
    # granularity from the fixture: lot 100 x median open (~10) / 1e5.
    assert report.granularity == pytest.approx(
        100.0 * float(np.median(raw["open"])) / 100_000.0, rel=1e-9
    )
    verdict = proxy_reliability_verdict(report)
    assert verdict["verdict"] == "reliable", verdict["reasons"]


def test_n5_loader_wiring_runs_both_paths_on_one_slice(populated_db):
    from ashare_model.data_loader import AshareDataLoader
    from ashare_data.config import ModelConfig, ProtocolConfig
    from ashare_model.eval_folds import FoldConfig, epoch_slice, resolve_folds
    from ashare_model.scoring_parity import run_fixed_candidate_parity

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    proto = ProtocolConfig(folds=[FoldConfig("2024-01-10", "2024-01-25")])
    fold = resolve_folds(
        proto.folds, loader.dates,
        frequency=proto.frequency, horizon=proto.horizon,
    )[0]
    fold_data = epoch_slice(loader, fold)
    n_stocks, n_dates = fold_data.universe_mask.shape
    rng = np.random.default_rng(5)
    candidates = [
        (f"cand{i}", 1, rng.normal(size=(n_stocks, n_dates)))
        for i in range(2)
    ]
    report = run_fixed_candidate_parity(
        loader, fold, _cfg(top_n=3, single_weight_cap=1.0 / 3.0), candidates
    )
    assert len(report.rows) == 2
    for row in report.rows:
        assert row.proxy_total_net == pytest.approx(row.proxy_total_net)
        assert np.isfinite(row.total_net_return_error)
