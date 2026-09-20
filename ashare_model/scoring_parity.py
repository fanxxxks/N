"""Scoring-vs-formal-execution parity measurement (p45, WP 9).

This module owns the *measurement and adjudication* of the gap between
the search-scoring path (``CandidateScorer`` -> the weight-space basket
``simulate_basket_daily_returns``) and the formal execution path
(``AshareBacktestEngine``: whole lots, share-level fees, cash fail-closed).
It changes when the parity *methodology* changes — not when either scored
path changes (neither path is modified by p45; the attribution probe found
zero class-1 inconsistencies, see ``docs/p45_scoring_engine_parity_draft.md``
§1).

The frozen tolerances are mechanism formulas of the lot granularity
``g = lot_size * price / capital``: a single leg's lot-floor error is at
most one lot, one rebalance touches at most ``2 * top_n`` legs, and the
fee delta comes from continuous-weight vs lot-floored notionals plus the
per-order minimum commission.  They scale linearly with ``g`` (coarse
lots at small capital -> proportionally looser bounds), and every report
records its own ``g`` so a verdict is always interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ashare_data.config import BacktestConfig
from ashare_data.processor import open_to_open_returns

SCORING_PARITY_VERSION = 1

# p45 §2 B2 frozen mechanism coefficients (measured anchors in the
# contract's §1 table: daily-net max 0.08g, fee max 0.011g, turnover max
# 1.6g on the canonical fixtures — each bound carries 2.5-4.5x headroom).
_DAILY_NET_COEFF = 0.2
_FEE_FRACTION_COEFF = 0.05
_TURNOVER_COEFF = 10.0
_TOTAL_NET_COEFF_PER_PERIOD = 0.05


def granularity(*, lot_size: int, price_ref: float, capital: float) -> float:
    """The one-lot weight granularity ``lot_size * price_ref / capital``."""

    if int(lot_size) < 1:
        raise ValueError(f"lot_size must be >= 1, got {lot_size}")
    if not float(price_ref) > 0.0:
        raise ValueError(f"price_ref must be > 0, got {price_ref}")
    if not float(capital) > 0.0:
        raise ValueError(f"capital must be > 0, got {capital}")
    return int(lot_size) * float(price_ref) / float(capital)


def parity_bounds(
    cfg: BacktestConfig, *, price_ref: float, n_periods: int
) -> dict[str, float]:
    """The p45 §2 B2 tolerance set at this configuration's granularity."""

    g = granularity(
        lot_size=int(getattr(cfg, "lot_size", 100)),
        price_ref=price_ref,
        capital=float(cfg.initial_capital),
    )
    return {
        "daily_net": _DAILY_NET_COEFF * g,
        "fee_fraction": _FEE_FRACTION_COEFF * g,
        "turnover": _TURNOVER_COEFF * g,
        "total_net": (
            _TOTAL_NET_COEFF_PER_PERIOD * g * max(int(n_periods), 0)
        ),
    }


def _total_return(daily: Sequence[float]) -> float:
    equity = 1.0
    for value in daily:
        equity *= 1.0 + float(value)
    return equity - 1.0


def ranking_comparison(
    proxy_scores: Mapping[str, float],
    engine_scores: Mapping[str, float],
    *,
    material_gap: float,
) -> tuple[float, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Spearman rho plus the material/tie inversion split of two rankings.

    An inversion is a pair the two paths order differently; it is
    *material* when the proxy score gap exceeds ``material_gap`` (the
    approximation cannot plausibly explain it) and a *tie* otherwise
    (disclosed, never counted against reliability).
    """

    names = sorted(proxy_scores)
    if set(names) != set(engine_scores):
        raise ValueError("proxy and engine scores must cover the same names")
    n = len(names)
    if n < 2:
        return 1.0, (), ()
    proxy_order = sorted(names, key=lambda name: (-float(proxy_scores[name]), name))
    engine_rank = {
        name: rank
        for rank, name in enumerate(
            sorted(names, key=lambda name: (-float(engine_scores[name]), name))
        )
    }
    d2 = sum(
        (rank - engine_rank[name]) ** 2
        for rank, name in enumerate(proxy_order)
    )
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    material: list[tuple[str, str]] = []
    tie: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = proxy_order[i], proxy_order[j]  # proxy ranks a above b
            if engine_rank[b] < engine_rank[a]:
                pair = (a, b)
                gap = abs(float(proxy_scores[a]) - float(proxy_scores[b]))
                (material if gap > float(material_gap) else tie).append(pair)
    return rho, tuple(material), tuple(tie)


@dataclass(frozen=True)
class CandidateParityRow:
    """One fixed candidate's proxy-vs-engine error report."""

    candidate: str
    direction: int
    proxy_direction: int
    engine_direction: int
    daily_net_error_max: float
    daily_net_error_mean: float
    fee_fraction_error_max: float
    turnover_error_max: float
    total_net_return_error: float
    total_fee_error: float
    proxy_total_net: float
    engine_total_net: float


@dataclass(frozen=True)
class ScoringParityReport:
    """The fixed-candidate comparison over one window and configuration."""

    granularity: float
    bounds: Mapping[str, float]
    rows: tuple[CandidateParityRow, ...]
    direction_flips: tuple[str, ...]
    ranking_spearman: float
    inversions_material: tuple[tuple[str, str], ...]
    inversions_tie: tuple[tuple[str, str], ...]


def _reference_price(open_: np.ndarray) -> float:
    positive = open_[open_ > 0.0]
    if positive.size == 0:
        raise ValueError("no positive open price to derive the granularity")
    return float(np.median(positive))


def _run_orientation(
    signal: np.ndarray,
    target: np.ndarray,
    raw_cache: Mapping[str, np.ndarray],
    ts_codes: Sequence[str],
    dates: Sequence[str],
    universe_mask: np.ndarray,
    cfg: BacktestConfig,
    *,
    blocked_buy: np.ndarray | None,
    blocked_sell: np.ndarray | None,
    rebalance_mask: np.ndarray | None,
):
    from .backtest import AshareBacktestEngine
    from .reward import simulate_basket_daily_returns

    sim = simulate_basket_daily_returns(
        signal,
        target,
        cfg,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        universe_mask=universe_mask,
        tie_break_keys=np.asarray(ts_codes),
        rebalance_mask=rebalance_mask,
    )
    engine = AshareBacktestEngine(cfg).run(
        signal,
        dict(raw_cache),
        list(ts_codes),
        list(dates),
        universe_mask=universe_mask,
        rebalance_mask=rebalance_mask,
    )
    return sim, engine


def compare_candidate_parity(
    candidates: Sequence[tuple[str, int, np.ndarray]],
    raw_cache: Mapping[str, np.ndarray],
    ts_codes: Sequence[str],
    dates: Sequence[str],
    *,
    universe_mask: np.ndarray,
    backtest_config: BacktestConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    rebalance_mask: np.ndarray | None = None,
) -> ScoringParityReport:
    """Compare fixed candidates through BOTH paths on one shared context.

    Every candidate ``(name, direction, signal)`` is run through the
    weight-space basket (the search-scoring proxy) and the formal engine
    on the **same window, masks, direction and configuration**, in both
    orientations so the direction each path would choose is measured
    (``proxy_direction`` vs ``engine_direction``; a disagreement is a
    direction flip and escalates to the research decision, p45 B3).
    The error families of the row use the candidate's supplied direction.
    """

    from .backtest import AshareBacktestEngine  # noqa: F401  (parity guard)

    open_ = np.asarray(raw_cache["open"], dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    n_stocks, n_dates = universe_mask.shape
    if open_.shape != (n_stocks, n_dates):
        raise ValueError(
            f"open shape {open_.shape} does not match universe_mask "
            f"{(n_stocks, n_dates)}"
        )
    target = open_to_open_returns(open_)
    price_ref = _reference_price(open_)

    rows: list[CandidateParityRow] = []
    proxy_scores: dict[str, float] = {}
    engine_scores: dict[str, float] = {}
    direction_flips: list[str] = []
    n_periods = 0
    for name, direction, signal in candidates:
        signal = np.asarray(signal, dtype=np.float64)
        if signal.shape != (n_stocks, n_dates):
            raise ValueError(
                f"candidate {name!r} signal shape {signal.shape} does not "
                f"match the window {(n_stocks, n_dates)}"
            )
        by_orientation = {
            orient: _run_orientation(
                float(orient) * signal,
                target,
                raw_cache,
                ts_codes,
                dates,
                universe_mask,
                backtest_config,
                blocked_buy=blocked_buy,
                blocked_sell=blocked_sell,
                rebalance_mask=rebalance_mask,
            )
            for orient in (1, -1)
        }
        proxy_direction = max(
            (1, -1),
            key=lambda o: _total_return(
                by_orientation[o][0].daily_net_returns
            ),
        )
        engine_direction = max(
            (1, -1),
            key=lambda o: _total_return(by_orientation[o][1].daily_returns),
        )
        if proxy_direction != engine_direction:
            direction_flips.append(str(name))
        sim, engine = by_orientation[int(direction)]
        sim_daily = np.asarray(sim.daily_net_returns, dtype=np.float64)
        eng_daily = np.asarray(engine.daily_returns, dtype=np.float64)
        n = min(sim_daily.size, eng_daily.size)
        n_periods = max(n_periods, n)
        net_err = np.abs(sim_daily[:n] - eng_daily[:n])
        sim_fee = np.asarray(sim.daily_cost_fractions, dtype=np.float64)
        eng_fee = np.asarray(engine.cost_fractions, dtype=np.float64)
        m = min(sim_fee.size, eng_fee.size)
        fee_err = np.abs(sim_fee[:m] - eng_fee[:m])
        sim_turn = np.asarray(sim.turnover, dtype=np.float64)
        eng_turn = np.asarray(engine.turnover, dtype=np.float64)
        k = min(sim_turn.size, eng_turn.size)
        turn_err = np.abs(sim_turn[:k] - eng_turn[:k])
        proxy_total = _total_return(sim_daily)
        engine_total = _total_return(eng_daily)
        rows.append(
            CandidateParityRow(
                candidate=str(name),
                direction=int(direction),
                proxy_direction=int(proxy_direction),
                engine_direction=int(engine_direction),
                daily_net_error_max=float(net_err.max()) if n else 0.0,
                daily_net_error_mean=float(net_err.mean()) if n else 0.0,
                fee_fraction_error_max=float(fee_err.max()) if m else 0.0,
                turnover_error_max=float(turn_err.max()) if k else 0.0,
                total_net_return_error=abs(proxy_total - engine_total),
                total_fee_error=abs(float(sim_fee[:m].sum() - eng_fee[:m].sum())),
                proxy_total_net=proxy_total,
                engine_total_net=engine_total,
            )
        )
        proxy_scores[str(name)] = proxy_total
        engine_scores[str(name)] = engine_total

    bounds = parity_bounds(
        backtest_config, price_ref=price_ref, n_periods=n_periods
    )
    rho, material, tie = ranking_comparison(
        proxy_scores,
        engine_scores,
        material_gap=bounds["daily_net"],
    )
    return ScoringParityReport(
        granularity=granularity(
            lot_size=int(getattr(backtest_config, "lot_size", 100)),
            price_ref=price_ref,
            capital=float(backtest_config.initial_capital),
        ),
        bounds=MappingProxyType(dict(bounds)),
        rows=tuple(rows),
        direction_flips=tuple(direction_flips),
        ranking_spearman=float(rho),
        inversions_material=material,
        inversions_tie=tie,
    )


def proxy_reliability_verdict(
    report: ScoringParityReport,
) -> dict[str, Any]:
    """The p45 B3 gate: may the proxy scores still rank this candidate set?

    ``reliable`` only when every error family is within the frozen
    bounds, no direction flipped and no material ranking inversion
    occurred.  Anything else is ``unreliable`` with itemized reasons —
    the shortlist must then be ranked by formal-engine re-verification
    (this module's comparison IS that re-verification harness).
    """

    bounds = dict(report.bounds)
    reasons: list[str] = []
    for row in report.rows:
        if row.daily_net_error_max > bounds["daily_net"]:
            reasons.append(
                f"{row.candidate}: daily net error {row.daily_net_error_max:.3e} "
                f"exceeds the parity bound {bounds['daily_net']:.3e}"
            )
        if row.fee_fraction_error_max > bounds["fee_fraction"]:
            reasons.append(
                f"{row.candidate}: fee fraction error "
                f"{row.fee_fraction_error_max:.3e} exceeds the parity bound "
                f"{bounds['fee_fraction']:.3e}"
            )
        if row.turnover_error_max > bounds["turnover"]:
            reasons.append(
                f"{row.candidate}: turnover error {row.turnover_error_max:.3e} "
                f"exceeds the parity bound {bounds['turnover']:.3e}"
            )
        if row.total_net_return_error > bounds["total_net"]:
            reasons.append(
                f"{row.candidate}: total net error "
                f"{row.total_net_return_error:.3e} exceeds the parity bound "
                f"{bounds['total_net']:.3e}"
            )
    for name in report.direction_flips:
        reasons.append(
            f"{name}: direction flip (proxy and engine choose opposite "
            "directions) - approximation has reached the direction "
            "decision, a research-decision item"
        )
    # Defence in depth: a row whose own orientations disagree is a flip
    # even if the aggregate list missed it (genuine reports populate both
    # from the same measurement; hand-assembled ones stay honest too).
    for row in report.rows:
        if row.proxy_direction != row.engine_direction:
            flip_reason = (
                f"{row.candidate}: direction flip (proxy and engine choose "
                "opposite directions) - approximation has reached the "
                "direction decision, a research-decision item"
            )
            if flip_reason not in reasons:
                reasons.append(flip_reason)
    for a, b in report.inversions_material:
        reasons.append(
            f"ranking: material inversion ({a}, {b}) - the proxy order and "
            "the engine order disagree beyond the parity bound"
        )
    return {
        "verdict": "reliable" if not reasons else "unreliable",
        "reasons": reasons,
        "scoring_parity_version": SCORING_PARITY_VERSION,
        "granularity": report.granularity,
        "bounds": dict(report.bounds),
        "direction_flips": list(report.direction_flips),
        "inversions_material": [list(p) for p in report.inversions_material],
        "inversions_tie": [list(p) for p in report.inversions_tie],
        "ranking_spearman": report.ranking_spearman,
    }


def run_fixed_candidate_parity(
    loader,
    fold,
    backtest_config: BacktestConfig,
    candidates: Sequence[tuple[str, int, np.ndarray]],
) -> ScoringParityReport:
    """Loader-level wiring: both paths consume the same epoch slice.

    The fold's test window (PIT universe mask, rebalance schedule, raw
    price cache, tradability matrices) is sliced once and handed to both
    paths, so the parity comparison cannot drift on window or mask
    alignment (p45 B1).
    """

    from .eval_folds import epoch_slice

    fold_data = epoch_slice(loader, fold)
    contract = fold.contract
    blocked_buy_full, blocked_sell_full = loader.tradability_masks()
    blocked_buy = blocked_buy_full[:, contract.test_signal_start : contract.test_price_end]
    blocked_sell = blocked_sell_full[
        :, contract.test_signal_start : contract.test_price_end
    ]
    return compare_candidate_parity(
        candidates,
        fold_data.raw,
        list(loader.ts_codes),
        fold_data.dates,
        universe_mask=fold_data.universe_mask,
        backtest_config=backtest_config,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        rebalance_mask=fold_data.rebalance_mask,
    )
