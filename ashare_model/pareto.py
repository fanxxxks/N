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
other).  :class:`IncrementalNonDominatedFront` maintains front 0 in
O(|front|) per arriving candidate for callers (the RL loop) that would
otherwise re-sort the whole accumulated pool per step.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def is_measurable(
    item: Any,
    objectives: Sequence[tuple[str, int]],
) -> bool:
    """Whether every objective of ``item`` is non-NaN.

    ``nondominated_fronts`` sorts items with a NaN objective to the last
    front (an unmeasurable objective never wins a selection); the
    incremental front keeps the same classification for items it never
    admits.  ``value == value`` is the NaN test (``inf`` stays measurable).
    """

    return all(
        _value(item, key) == _value(item, key) for key, _ in objectives
    )


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

    measured = [item for item in pool if is_measurable(item, objectives)]
    unmeasured = [item for item in pool if not is_measurable(item, objectives)]

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


class IncrementalNonDominatedFront:
    """First Pareto front maintained incrementally (P1-1).

    ``nondominated_fronts`` re-sorts the whole pool, which costs
    O(N^2) domination comparisons (worse across the front peeling)
    per call; the RL loop used to pay that on every step against the
    full accumulated candidate pool.  This class keeps only the current
    first front and folds each arriving candidate in with O(|front|)
    comparisons:

    * a new candidate is dominated iff some front member dominates it —
      dominance is transitive, so any dominator outside the front would
      imply one inside it;
    * the only members a new candidate can remove from the front are
      those it dominates itself; every removed member is then dominated
      by a member that stays (the newcomer), so the invariant "every
      pool item outside the front is dominated by a front member"
      survives each insertion.

    Membership therefore stays exactly the first front
    :func:`nondominated_fronts` computes over the same accumulated set
    (pinned by tests with random batches against the one-shot sort).
    Items with a NaN objective are classified ``"unmeasurable"`` and
    never admitted, mirroring the last-front rule of the one-shot sort.

    ``add`` must see each pool key at most once per pool generation.
    Re-adding a key that currently sits in the front (a replaced score)
    can release previously dominated candidates this structure no longer
    holds, so it sets :attr:`dirty` — the caller then rebuilds from the
    authoritative pool.  Scores are deterministic functions of the token
    key, so the flag is a never-taken safety valve in the trainer.

    ``dominance_comparisons`` counts every domination comparison per
    :meth:`add` call: the performance contract ("per-step cost follows
    the batch and the front size, not the accumulated pool") is pinned
    by counting instead of wall-clock (CI-stable).
    """

    def __init__(
        self, objectives: Sequence[tuple[str, int]]
    ) -> None:
        self._objectives = tuple(objectives)
        self._members: dict[Any, Any] = {}
        self.dirty = False
        self.dominance_comparisons = 0

    def __len__(self) -> int:
        return len(self._members)

    def keys(self) -> list[Any]:
        """Identity keys of the current front members (insertion order)."""

        return list(self._members)

    def members(self) -> list[Any]:
        """Current first-front members (insertion order)."""

        return list(self._members.values())

    def reset(self) -> None:
        """Drop all state (a caller-requested rebuild starts clean)."""

        self._members.clear()
        self.dirty = False

    def add(self, key: Any, item: Any) -> str:
        """Fold ``item`` (identity ``key``) into the front.

        Returns ``"unmeasurable"`` (NaN objective — never admitted),
        ``"dominated"`` (some front member dominates it) or ``"front"``
        (admitted; members it dominates were removed).
        """

        if not is_measurable(item, self._objectives):
            return "unmeasurable"
        if key in self._members:
            # A replaced item changes domination outcomes the incremental
            # structure cannot replay — ask the caller for a rebuild.
            del self._members[key]
            self.dirty = True
        for other in self._members.values():
            self.dominance_comparisons += 1
            if _dominates(other, item, self._objectives):
                return "dominated"
        kept: dict[Any, Any] = {}
        for member_key, member in self._members.items():
            self.dominance_comparisons += 1
            if not _dominates(item, member, self._objectives):
                kept[member_key] = member
        kept[key] = item
        self._members = kept
        return "front"
