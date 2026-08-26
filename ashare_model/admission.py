"""RL admission experiment logic (T2-03).

The admission question: does RL (REINFORCE policy) beat the search
baselines — uniform random and strongly-typed GP — under **identical
unique-semantic-evaluation budgets**, across at least five independent
initializations?  The pre-registered rule (recorded in the Phase-2
measurement log):

* per seed, every baseline receives exactly the RL run's actual
  ``unique_semantic_evals`` (the v18 budget ledger), so the comparisons
  are budget-fair;
* two metrics per seed: the normalized best-so-far area (average of the
  best validation reward over the whole budget) and the OOS active IR of
  the selected formula;
* RL is **admitted** when, against *both* random and GP, RL's median is
  strictly better on *both* metrics and RL wins at least
  ``win_fraction`` of the seeds on each metric (default 4 of 5).

If RL is not admitted, the default searcher flips to ``gp`` and RL stays
an opt-in experimental backend (``model.searcher: rl``).
"""

from __future__ import annotations

import math

# Independent policy initializations (>= 5): init_seed re-initializes the
# weights; the sampling seed stays fixed, so the runs differ only in the
# starting policy.
INIT_SEEDS = (42, 7, 2024, 1337, 999)

# Admission tier: a fraction of the protocol screening tier, run on a
# fixed window cap (the head slice of the fold's training window) so the
# five seeds × four searchers stay tractable.  Every searcher measures
# the exact same capped window with the exact same per-seed budget — the
# comparison is internal to the admission, so the cap is fair by
# construction.  The full-window screening tier would cost ~3 s per
# unique evaluation on this machine (measured), i.e. ~26 h for one
# screening-tier fold; the cap is the documented tractable tier.
ADMISSION_STEPS = 8
ADMISSION_BATCH = 128
ADMISSION_WINDOW = (300, 400)  # (stocks, dates) head slice of the fold window

# Fraction of seeds RL must win on each metric against each baseline.
WIN_FRACTION = 0.8


def best_so_far_curve_values(
    curve: list[tuple[float, float]], budget: int
) -> list[float]:
    """Best-so-far value at every integer budget point ``1..budget``.

    ``curve`` is ``[(cumulative_budget, best_reward), ...]`` (monotone in
    both coordinates).  The value at budget ``b`` is the best reward
    achieved by any evaluation with ``cumulative_budget <= b`` — the step
    function the searcher actually delivered.
    """

    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    values: list[float] = []
    best = -math.inf
    index = 0
    for b in range(1, budget + 1):
        while index < len(curve) and curve[index][0] <= b:
            best = max(best, float(curve[index][1]))
            index += 1
        values.append(best)
    return values


def best_so_far_area(curve: list[tuple[float, float]], budget: int) -> float:
    """Normalized best-so-far area: the mean of the best reward over the
    whole budget (== area under the step function divided by the budget).
    Comparable across searchers only at equal budgets."""

    values = best_so_far_curve_values(curve, budget)
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("-inf")
    return float(sum(finite) / len(finite))


def _wins(rl_values: list[float], baseline_values: list[float]) -> tuple[int, int]:
    wins = sum(1 for a, b in zip(rl_values, baseline_values) if a > b)
    return wins, len(rl_values)


def decide_admission(
    *,
    rl_areas: list[float],
    rl_oos_irs: list[float],
    baseline_areas: dict[str, list[float]],
    baseline_oos_irs: dict[str, list[float]],
    win_fraction: float = WIN_FRACTION,
) -> dict:
    """Apply the pre-registered admission rule.

    ``baseline_areas`` / ``baseline_oos_irs`` map baseline names (e.g.
    ``random``, ``gp``) to their per-seed values, aligned with the RL
    lists by seed.  Returns the verdict with the per-baseline evidence.
    """

    if len(set(map(len, [rl_areas, rl_oos_irs, *baseline_areas.values(),
                         *baseline_oos_irs.values()]))) != 1:
        raise ValueError("all per-seed lists must be aligned")
    n_seeds = len(rl_areas)
    min_wins = math.ceil(win_fraction * n_seeds)
    rl_med_area = float(sorted(rl_areas)[n_seeds // 2])
    rl_med_ir = float(sorted(rl_oos_irs)[n_seeds // 2])

    evidence: dict[str, dict] = {}
    admitted = True
    for name in baseline_areas:
        wins_area, _ = _wins(rl_areas, baseline_areas[name])
        wins_ir, _ = _wins(rl_oos_irs, baseline_oos_irs[name])
        med_area = float(sorted(baseline_areas[name])[n_seeds // 2])
        med_ir = float(sorted(baseline_oos_irs[name])[n_seeds // 2])
        beats = (
            rl_med_area > med_area
            and rl_med_ir > med_ir
            and wins_area >= min_wins
            and wins_ir >= min_wins
        )
        evidence[name] = {
            "wins_area": wins_area,
            "wins_oos_ir": wins_ir,
            "min_wins": min_wins,
            "rl_median_area": rl_med_area,
            "baseline_median_area": med_area,
            "rl_median_oos_ir": rl_med_ir,
            "baseline_median_oos_ir": med_ir,
            "rl_dominates": beats,
        }
        admitted = admitted and beats

    return {
        "n_seeds": n_seeds,
        "win_fraction": win_fraction,
        "min_wins": min_wins,
        "admitted": admitted,
        "default_searcher": "rl" if admitted else "gp",
        "evidence": evidence,
    }


def default_searcher(verdict: dict) -> str:
    """The production default searcher per the admission verdict."""

    return verdict["default_searcher"]
