"""Pre-registered paired admission rules for the experimental RL searcher.

P4 compares imitation-initialized RL with random-initialized RL and GP under
the same unique-semantic-evaluation request in at least five independent seed
pairs.  Both best-so-far area and OOS active IR must dominate by the fixed
median and paired-win rule.  Missing evidence rejects admission; it is never
dropped, imputed, or replaced by another pair.
"""

from __future__ import annotations

import math

# P4 pairs: the same seed initializes and samples every arm within one pair;
# different rows use different seeds.  This replaces T2's fixed baseline
# seeds, which repeated one GP/TPE/Random result five times.
PAIR_SEEDS = (42, 7, 2024, 1337, 999)
ADMISSION_RULE_VERSION = 2
_PAIRED_BACKENDS = ("gp", "tpe", "random", "rl_random", "rl_imitation")

# Admission tier: a fraction of the protocol screening tier, run on a
# fixed window cap (the head slice of the fold's training window) so the
# five seeds × five searchers stay tractable.  Every searcher measures
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


def paired_seed_plan(
    seeds: tuple[int, ...] = PAIR_SEEDS,
) -> tuple[dict[str, object], ...]:
    """Auditable seed plan: one common seed per independent pair."""

    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < 5:
        raise ValueError("paired admission requires at least five seeds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("paired admission seeds must be unique")
    return tuple(
        {
            "pair_index": index,
            "pair_seed": seed,
            "backend_seeds": {backend: seed for backend in _PAIRED_BACKENDS},
        }
        for index, seed in enumerate(normalized)
    )


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

    if set(baseline_areas) != set(baseline_oos_irs):
        raise ValueError("baseline metric names must be aligned")
    if len(set(map(len, [rl_areas, rl_oos_irs, *baseline_areas.values(),
                         *baseline_oos_irs.values()]))) != 1:
        raise ValueError("all per-seed lists must be aligned")
    n_seeds = len(rl_areas)
    if n_seeds == 0:
        raise ValueError("per-seed lists must not be empty")
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


def decide_p4_admission(
    *,
    imitation_areas: list[float | None],
    imitation_oos_irs: list[float | None],
    random_rl_areas: list[float | None],
    random_rl_oos_irs: list[float | None],
    gp_areas: list[float | None],
    gp_oos_irs: list[float | None],
    win_fraction: float = WIN_FRACTION,
) -> dict:
    """P4 rule: imitation RL must dominate random-init RL and GP.

    Missing/non-finite measurements are an automatic, explicit rejection;
    they are never dropped from a pair or imputed from another backend.
    """

    series = {
        "imitation_areas": imitation_areas,
        "imitation_oos_irs": imitation_oos_irs,
        "random_rl_areas": random_rl_areas,
        "random_rl_oos_irs": random_rl_oos_irs,
        "gp_areas": gp_areas,
        "gp_oos_irs": gp_oos_irs,
    }
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1:
        raise ValueError("all paired metric lists must be aligned")
    n_pairs = next(iter(lengths), 0)
    if n_pairs < 5:
        raise ValueError("P4 admission requires at least five aligned pairs")
    invalid_metrics = [
        f"{name}[{index}]"
        for name, values in series.items()
        for index, value in enumerate(values)
        if value is None or not math.isfinite(float(value))
    ]
    if invalid_metrics:
        return {
            "rule_version": ADMISSION_RULE_VERSION,
            "n_pairs": n_pairs,
            "win_fraction": win_fraction,
            "min_wins": math.ceil(win_fraction * n_pairs),
            "rl_admitted": False,
            "advanced_rl_allowed": False,
            "default_searcher": "gp",
            "invalid_metrics": invalid_metrics,
            "evidence": {},
        }

    generic = decide_admission(
        rl_areas=[float(value) for value in imitation_areas],
        rl_oos_irs=[float(value) for value in imitation_oos_irs],
        baseline_areas={
            "random_rl": [float(value) for value in random_rl_areas],
            "gp": [float(value) for value in gp_areas],
        },
        baseline_oos_irs={
            "random_rl": [float(value) for value in random_rl_oos_irs],
            "gp": [float(value) for value in gp_oos_irs],
        },
        win_fraction=win_fraction,
    )
    admitted = bool(generic["admitted"])
    return {
        "rule_version": ADMISSION_RULE_VERSION,
        "n_pairs": n_pairs,
        "win_fraction": win_fraction,
        "min_wins": generic["min_wins"],
        "rl_admitted": admitted,
        "advanced_rl_allowed": admitted,
        "default_searcher": "rl" if admitted else "gp",
        "invalid_metrics": [],
        "evidence": generic["evidence"],
    }


def apply_p4_tier_gate(
    metric_verdict: dict,
    *,
    steps: int,
    batch_size: int,
    window: tuple[int, int],
) -> dict:
    """Prevent budget/window overrides from becoming promotion evidence."""

    registered = (
        int(steps) == ADMISSION_STEPS
        and int(batch_size) == ADMISSION_BATCH
        and tuple(int(value) for value in window) == ADMISSION_WINDOW
    )
    metric_rule_passed = bool(metric_verdict.get("rl_admitted", False))
    blockers: list[str] = []
    if not registered:
        blockers.append("non_registered_admission_tier")
    if not metric_rule_passed:
        blockers.append("metric_rule_failed")

    verdict = dict(metric_verdict)
    verdict.update(
        metric_rule_passed=metric_rule_passed,
        registered_tier=registered,
        promotion_blockers=blockers,
    )
    if blockers:
        verdict.update(
            rl_admitted=False,
            advanced_rl_allowed=False,
            default_searcher="gp",
        )
    return verdict


def default_searcher(verdict: dict) -> str:
    """The production default searcher per the admission verdict."""

    return verdict["default_searcher"]
