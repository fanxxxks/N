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

p53 (WP16-B, ``reports/wp16/FROZEN_B.md``) — P45 comparability: the
harness threads one explicit ``local_signal_range`` through BOTH paths,
refuses (never trims) length/date/cache-axis mismatches before any error
is computed, meters the basket-vs-engine blocked-buy asymmetry per row
(``blocked_extra_buy_columns``), and names its orientation diagnostics
under the §4 contract: ``proxy_direction``/``engine_direction`` are the
TEST-WINDOW CUMULATIVE-NET-RETURN ORIENTATION (a diagnostic of this
harness), not the production Rule A validation-reward orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ashare_data.config import BacktestConfig
from ashare_data.processor import (
    exact_tradability_matrix,
    open_to_open_returns,
)

SCORING_PARITY_VERSION = 1

# p53 (WP16-B, reports/wp16/FROZEN_B.md §4.2): the frozen disclosure
# string every output surface that names proxy_direction /
# engine_direction / direction_flips must carry.  These fields are the
# TEST-WINDOW CUMULATIVE-NET-RETURN ORIENTATION diagnostic (each path is
# run in both orientations and the higher test-window total wins) — a
# diagnostic internal to this harness, explicitly NOT the production
# Rule A validation-reward orientation (candidates.py).
ORIENTATION_DISCLOSURE = (
    "proxy/engine_direction = 测试窗累计净收益择向（诊断），"
    "非生产 Rule A 验证 reward 择向"
)

# p53 (WP16-B FROZEN_B §1 B1.2/B4.2): the p19 price-semantics keys the
# harness itself consumes or delegates to the engine; a cache missing
# any of them is refused BEFORE any comparison runs (fail-closed on the
# label basis and the constraint tables, exactly like the engine).
_REQUIRED_CACHE_KEYS = (
    "open",
    "adjusted_open",
    "up_limit",
    "down_limit",
    "suspended",
)

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
    # p53 (WP16-B FROZEN_B §1 F5): explicit metering of the blocking
    # asymmetry — the number of date columns where the basket's
    # blocked_buy blocks a name the engine's own raw-cache matrix does
    # not (the loader's tradability_masks additionally folds in
    # ~universe_mask and adj-factor demotion).  0 when the two blocking
    # sets coincide (B5.1) or no basket mask was supplied.
    blocked_extra_buy_columns: int = 0


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
    signal_range: tuple[int, int] | None = None,
):
    from .backtest import AshareBacktestEngine
    from .reward import simulate_basket_daily_returns

    # p53 (WP16-B FROZEN_B §1 F3/B3.1): the ONE window value is handed
    # to both paths — the basket's signal_range and the engine's
    # signal_range are the same tuple, never two independently defaulted
    # windows (the defaults fork when execution_delay > 1).
    sim = simulate_basket_daily_returns(
        signal,
        target,
        cfg,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        universe_mask=universe_mask,
        tie_break_keys=np.asarray(ts_codes),
        rebalance_mask=rebalance_mask,
        signal_range=signal_range,
    )
    engine = AshareBacktestEngine(cfg).run(
        signal,
        dict(raw_cache),
        list(ts_codes),
        list(dates),
        universe_mask=universe_mask,
        rebalance_mask=rebalance_mask,
        signal_range=signal_range,
    )
    return sim, engine


def _reject_length_mismatch(
    family: str, n_sim: int, n_eng: int, candidate: str, window_id: str
) -> None:
    """p53 (WP16-B FROZEN_B §1 B4.1): refuse, never trim to the shorter.

    The frozen four message elements: the semantic tag
    ``"length mismatch"``, ``sim={n_sim}``, ``engine={n_eng}`` and the
    candidate/window identifier.  The rejection fires BEFORE any error
    of the family is computed (B4.4: no partial rows, no partial file).
    """

    raise ValueError(
        f"length mismatch: {family} sim={n_sim} engine={n_eng} for "
        f"candidate {candidate!r} (window={window_id}); refusing to trim "
        "to the shorter side (WP16-B FROZEN_B §1 B4.1)"
    )


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
    local_signal_range: tuple[int, int] | None = None,
) -> ScoringParityReport:
    """Compare fixed candidates through BOTH paths on one shared context.

    Every candidate ``(name, direction, signal)`` is run through the
    weight-space basket (the search-scoring proxy) and the formal engine
    on the **same window, masks, direction and configuration**.  The
    error families of the row use the candidate's supplied direction
    (F6: ±1 is only a signal multiplier).

    Diagnostic naming (p53, WP16-B FROZEN_B §4): ``proxy_direction`` /
    ``engine_direction`` are the **test-window cumulative-net-return
    orientation** — each path is additionally run in both orientations
    and the side with the higher test-window cumulative net return is
    reported (a disagreement between the two PATHS' orderings is the
    direction-flip diagnostic).  This is a diagnostic of this harness
    only; it is NOT the production Rule A validation-reward orientation
    (candidates.py) and must never be presented as such.

    ``local_signal_range`` (p53, FROZEN_B §1 F3) is the fold-local test
    window ``(s0, s1)`` of signal columns, handed to BOTH paths as their
    ``signal_range``.  ``None`` keeps each path's own default (backward
    compatible); panel reruns must pass it explicitly.  A window whose
    last signal lacks a complete ``t + delay + 1`` exit is refused
    before anything runs (B3.3/B3.5, the engine's own exit semantics).

    Input consistency (FROZEN_B §1 F4/B4.2): the date axis must match
    the window columns exactly, carry no duplicates, and every 2-D
    ``raw_cache`` entry must match the window shape — a mismatch of one
    column anywhere is a ``ValueError``, never a quiet comparison of
    two different axes.  Output consistency (B4.1): the two paths' daily
    net / cost / turnover series must be the SAME length, else
    ``ValueError`` — trimming to the shorter side is forbidden.
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
    # --- p53 (WP16-B FROZEN_B §1 B4.2): the actual date axis. --------
    dates = list(dates)
    if len(dates) != n_dates:
        raise ValueError(
            f"date axis length {len(dates)} does not match the window "
            f"columns {n_dates} (universe_mask/open); refusing to compare "
            "two different axes (WP16-B FROZEN_B §1 B4.2)"
        )
    if len(set(dates)) != len(dates):
        raise ValueError(
            f"date axis contains duplicates: {len(dates)} entries, "
            f"{len(set(dates))} distinct — the session identity is "
            "ambiguous (WP16-B FROZEN_B §1 B4.2)"
        )
    # --- p53 (WP16-B FROZEN_B §1 B1.2): fail closed on the p19 keys. --
    missing_keys = [
        key for key in _REQUIRED_CACHE_KEYS if key not in raw_cache
    ]
    if missing_keys:
        raise ValueError(
            "raw_cache is missing the p19 price-semantics keys "
            f"{missing_keys}: the label basis is adjusted_open and the "
            "engine needs the exact constraint tables; refusing before "
            "any comparison (WP16-B FROZEN_B §1 B1.2)"
        )
    for key in _REQUIRED_CACHE_KEYS:
        array = np.asarray(raw_cache[key])
        if array.ndim == 2 and array.shape != (n_stocks, n_dates):
            raise ValueError(
                f"raw_cache[{key!r}] shape {array.shape} does not match "
                f"the window {(n_stocks, n_dates)}: the two paths would "
                "consume different price axes (WP16-B FROZEN_B §1 B4.2)"
            )
    # --- p53 (WP16-B FROZEN_B §1 F3/B3.3/B3.5): the shared window. ---
    if local_signal_range is not None:
        try:
            s0, s1 = int(local_signal_range[0]), int(local_signal_range[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                "local_signal_range must be an (s0, s1) int pair, got "
                f"{local_signal_range!r}"
            ) from exc
        if s0 < 0 or s1 < s0:
            raise ValueError(
                f"local_signal_range {(s0, s1)} is not a well-formed "
                "half-open window"
            )
        delay = 1  # the engine's default execution delay (never overridden here)
        for t in range(s0, s1):
            if t + delay + 1 >= n_dates:
                raise ValueError(
                    f"local_signal_range {(s0, s1)} includes signal t={t} "
                    f"without a complete t+{delay + 1} exit: "
                    f"t+{delay + 1}={t + delay + 1} >= n_dates={n_dates}; "
                    f"the price tail must carry at least "
                    f"s1+{delay + 1} <= n_dates columns "
                    "(WP16-B FROZEN_B §1 B3.3/B3.5)"
                )
        window_id = f"local_signal_range={(s0, s1)}"
    else:
        window_id = f"default(0, {max(n_dates - 2, 0)})"
    # Label basis = adjusted open (p19 §2.3), the same basis the engine
    # derives its target returns from (backtest.py: adjusted_open); with
    # raw nominal opens the two paths diverge systematically on every
    # ex-dividend/rights session (复权因子 != 1) and the parity error
    # families would measure the basis mismatch, not the approximation.
    target = open_to_open_returns(
        np.asarray(raw_cache["adjusted_open"], dtype=np.float64)
    )
    price_ref = _reference_price(open_)
    # p53 (WP16-B FROZEN_B §1 F5): the engine's own blocked-buy matrix,
    # derived exactly as the engine derives it inside run() — the
    # basket's mask comes from the loader's tradability_masks() and may
    # be strictly stronger; the per-row extra-column count makes that
    # asymmetry an explicit measurement instead of a hidden one.
    open_clean = np.nan_to_num(
        open_, nan=0.0, posinf=0.0, neginf=0.0
    )
    engine_blocked_buy, _ = exact_tradability_matrix(
        open_clean,
        np.asarray(raw_cache["up_limit"], dtype=np.float64),
        np.asarray(raw_cache["down_limit"], dtype=np.float64),
        np.asarray(raw_cache["suspended"]),
        "buy",
    )
    if blocked_buy is not None:
        basket_buy = np.asarray(blocked_buy, dtype=bool)
        blocked_extra_buy_columns = int(
            np.count_nonzero(np.any(basket_buy & ~engine_blocked_buy, axis=0))
        )
    else:
        blocked_extra_buy_columns = 0

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
                signal_range=local_signal_range,
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
        sim_fee = np.asarray(sim.daily_cost_fractions, dtype=np.float64)
        eng_fee = np.asarray(engine.cost_fractions, dtype=np.float64)
        sim_turn = np.asarray(sim.turnover, dtype=np.float64)
        eng_turn = np.asarray(engine.turnover, dtype=np.float64)
        # p53 (WP16-B FROZEN_B §1 B4.1): each family asserts its OWN
        # length equality and refuses on mismatch — no min-trim.
        n = sim_daily.size
        if n != eng_daily.size:
            _reject_length_mismatch(
                "daily net returns", n, eng_daily.size, str(name), window_id
            )
        m = sim_fee.size
        if m != eng_fee.size:
            _reject_length_mismatch(
                "cost fractions", m, eng_fee.size, str(name), window_id
            )
        k = sim_turn.size
        if k != eng_turn.size:
            _reject_length_mismatch(
                "turnover", k, eng_turn.size, str(name), window_id
            )
        n_periods = max(n_periods, n)
        net_err = np.abs(sim_daily - eng_daily)
        fee_err = np.abs(sim_fee - eng_fee)
        turn_err = np.abs(sim_turn - eng_turn)
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
                total_fee_error=abs(float(sim_fee.sum() - eng_fee.sum())),
                proxy_total_net=proxy_total,
                engine_total_net=engine_total,
                blocked_extra_buy_columns=blocked_extra_buy_columns,
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
    # p53 (WP16-B FROZEN_B §4): the flip wording speaks ONLY of the two
    # paths' test-window cumulative-net-return orderings (§4.4's allowed
    # form) — it never claims anything about the production Rule A
    # validation-reward orientation, which this harness does not measure.
    def _flip_reason(candidate: str, proxy: int, engine: int) -> str:
        return (
            f"{candidate}: direction-flip diagnostic — the two paths order "
            f"the two orientations oppositely on the test-window "
            f"cumulative-net-return basis (proxy {proxy:+d} / engine "
            f"{engine:+d}); a research-decision item, NOT the production "
            "Rule A validation-reward orientation"
        )

    flipped_from_rows = set()
    for row in report.rows:
        if row.proxy_direction != row.engine_direction:
            flipped_from_rows.add(row.candidate)
            reasons.append(
                _flip_reason(
                    row.candidate,
                    int(row.proxy_direction),
                    int(row.engine_direction),
                )
            )
    for name in report.direction_flips:
        # Defence in depth: a flip the rows themselves did not surface
        # (genuine reports populate both from the same measurement;
        # hand-assembled ones stay honest too).
        if name not in flipped_from_rows:
            row = next(
                (r for r in report.rows if r.candidate == name), None
            )
            reasons.append(
                _flip_reason(
                    name,
                    int(row.proxy_direction) if row else 0,
                    int(row.engine_direction) if row else 0,
                )
            )
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
        # p53 (WP16-B FROZEN_B §4.2): the frozen disclosure travels with
        # every verdict that names the orientation diagnostics.
        "orientation_disclosure": ORIENTATION_DISCLOSURE,
    }


def run_fixed_candidate_parity(
    loader,
    fold,
    backtest_config: BacktestConfig,
    candidates: Sequence[tuple[str, int, np.ndarray]],
    *,
    local_signal_range: tuple[int, int] | None = None,
) -> ScoringParityReport:
    """Loader-level wiring: both paths consume the same epoch slice.

    The fold's test window (PIT universe mask, rebalance schedule, raw
    price cache, tradability matrices) is sliced once and handed to both
    paths, so the parity comparison cannot drift on window or mask
    alignment (p45 B1).

    ``local_signal_range`` (p53, WP16-B FROZEN_B §1 F3) is forwarded
    verbatim to both paths.  ``None`` keeps the paths' own defaults;
    the fold-true window for the rerun protocol (FROZEN_B §3.2 second
    pass) is ``(0, contract.test_signal_end - contract.test_signal_start)``
    in the sliced frame's local columns.
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
        local_signal_range=local_signal_range,
    )
