"""Pareto-front selection (T1-04).

Eligibility (the hard constraints) filters candidates first; among the
eligible, selection prefers non-dominated candidates on the primary
portfolio objectives — active IR (maximize), risk exposure (minimize),
turnover (minimize), capacity utilization (minimize) — instead of
stuffing every objective into one fragile scalar.  The scalar reward
remains the RL gradient signal; Pareto ranking is the *selection*
policy.

``nondominated_fronts`` is the standard NSGA-II-style non-dominated
sorting: front 0 holds the candidates no other candidate beats on every
objective, front 1 holds those dominated only by front-0 members, etc.
Ties on all objectives land in the same front (none dominates the
other).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _value(item: Any, key: str) -> float:
    """Objective value of ``item`` (dataclass attribute or dict key)."""

    if isinstance(item, dict):
        return float(item[key])
    return float(getattr(item, key))


def _dominates(
    a: Any,
    b: Any,
    objectives: Sequence[tuple[str, int]],
) -> bool:
    """Whether ``a`` beats ``b`` on every objective and strictly on one."""

    strictly_better = False
    for key, direction in objectives:
        av, bv = _value(a, key), _value(b, key)
        if direction > 0:
            if av < bv:
                return False
            strictly_better = strictly_better or av > bv
        else:
            if av > bv:
                return False
            strictly_better = strictly_better or av < bv
    return strictly_better


def nondominated_fronts(
    items: Iterable[Any],
    objectives: Sequence[tuple[str, int]],
) -> list[list[Any]]:
    """Non-dominated fronts of ``items`` under ``objectives``.

    ``objectives`` is a list of ``(attribute, direction)`` pairs with
    ``direction`` +1 = maximize, -1 = minimize.  Front 0 = non-dominated;
    each later front is dominated only by members of earlier fronts.
    Items with non-finite objective values sort to the *last* front
    (unmeasurable objectives never win a selection).
    """

    pool = list(items)
    if not objectives or not pool:
        return [pool] if pool else []

    def measurable(item: Any) -> bool:
        return all(
            _value(item, key) == _value(item, key) for key, _ in objectives
        )

    measured = [item for item in pool if measurable(item)]
    unmeasured = [item for item in pool if not measurable(item)]

    fronts: list[list[Any]] = []
    remaining = list(measured)
    while remaining:
        front: list[Any] = []
        dominated: list[Any] = []
        for i, candidate in enumerate(remaining):
            if any(
                _dominates(other, candidate, objectives)
                for j, other in enumerate(remaining)
                if j != i
            ):
                dominated.append(candidate)
            else:
                front.append(candidate)
        fronts.append(front)
        remaining = dominated
    if unmeasured:
        fronts.append(unmeasured)
    return fronts
