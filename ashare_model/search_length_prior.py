"""P14 proposal length prior (search digest P1-1c) — single authority.

Contract: ``docs/p14_search_digest_preregistration.md`` §5.3.  Profile
``p14-uniform-2-11-v1``: the target proposal content-length distribution is
discrete-uniform over ``{2, …, max_formula_len − 1}`` (at the production
``max_len`` of 12: uniform over {2..11}, i.e. a 10% target share for the
cap length against the measured P10 stacking of ~91%).

Hard properties:

* **Space/cap untouched** — the prior only re-weights the samplers'
  proposal distributions; the legal token set, the EOS-inclusive length
  cap (``max_formula_len``), the budget accounting and the seeds are
  untouched (p14 §4/§5.3; AGENTS §7).
* **Degradation** — with fewer than 4 positions the profile is inactive
  (no termination targets, every per-position EOS probability 0.0) and the
  samplers keep the legacy uniform behaviour; legal sets without EOS fall
  through to the legacy behaviour.
* **Deterministic** — both consumers (the random sampler's per-position
  weights and the TPE per-trial length targets) are pure functions of the
  profile and the caller's seeded RNG, so a fixed seed reproduces the
  exact proposal stream.
* Any recalibration is a profile-id change plus a ``SEARCH_CONTRACT``
  bump (p14 §5.3/§6).

Implementation note (disclosed to t24): the TPE consumer honors a per-trial
target content length (drawn from :func:`sample_target_content_length`) by
forcing EOS at the first legal position at or after the target — Optuna's
TPESampler samples the categorical index space through its posterior (not
uniformly), so a slot-share remap of the raw index could not reliably
induce the preregistered distribution.  The preregistered *distribution*
and its acceptance bounds are unchanged and are pinned directly by
``tests/test_tpe_search.py``.
"""

from __future__ import annotations

import random
from typing import Sequence

LENGTH_PRIOR_PROFILE = "p14-uniform-2-11-v1"

#: Shortest content length the target distribution covers.
_MIN_CONTENT = 2


def eos_probability(step: int, max_formula_len: int) -> float:
    """Conditional ``P(EOS at this position | reached it)`` under the
    profile.

    Positions below ``_MIN_CONTENT`` never terminate (the target starts at
    two content tokens); the final position forces completion (q = 1) so
    every sequence stays inside the EOS-inclusive cap; profiles with fewer
    than four positions are inactive (degradation rule).
    """

    content_max = max_formula_len - 1
    levels = content_max - _MIN_CONTENT + 1
    if levels < 2:  # max_formula_len < 4: profile inactive (p14 §5.3)
        return 0.0
    if step < _MIN_CONTENT:
        return 0.0
    if step >= content_max:
        return 1.0
    target = 1.0 / levels
    survived = 1.0 - target * (step - _MIN_CONTENT)
    return target / survived


def sample_target_content_length(
    rng: random.Random, max_formula_len: int
) -> int | None:
    """Draw one target content length from the profile's discrete-uniform
    target distribution (``None`` when the profile is inactive).  The TPE
    consumer forces EOS at the first legal position at or after the drawn
    target, which induces the preregistered distribution over proposals."""

    content_max = max_formula_len - 1
    levels = content_max - _MIN_CONTENT + 1
    if levels < 2:  # max_formula_len < 4: profile inactive (p14 §5.3)
        return None
    return rng.randint(_MIN_CONTENT, content_max)
