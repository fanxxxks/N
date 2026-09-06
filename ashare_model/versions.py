"""Single home for version constants shared across the train/evaluation
boundary.

``PROTOCOL_VERSION`` used to be defined in ``evaluation.py``, which forced
``train.py`` to import it lazily (``evaluation`` → ``eval_search`` →
``train`` is a module-level edge, so ``train`` could not import
``evaluation`` at module level).  Moving the constant into this leaf
module (IP-07a) lets both hot-path modules import it at module level
without a cycle.  The value is unchanged — home-address migration only,
no version bump (AGENTS.md §3.2 version-impact table).

Constants with a single natural owner (reward, grammar, model, …) stay in
their owner modules; the artifact-facing owner index lives in
``ashare_model.runspec._VERSION_IMPORTS``.

p19 (``docs/p19_market_price_semantics_contract.md`` §5): "25" -> "26" —
the evaluation protocol's inputs changed (label price basis switched to
the adjusted series and tradability to the exact constraint tables).

p21 (``docs/p21_r1_domain_return_research_preregistration.md`` §8,
APPROVED 2026-09-06): "26" -> "27" — the protocol gains the additive
``protocol.tier_set`` run-level restriction and records ``tier_set`` /
``tier_restricted_features`` in the artifact; ``tier_set=None`` is
byte-identical to v26 behavior and v26 payloads stay legacy (version
mismatch, never converted).
"""

PROTOCOL_VERSION = "27"
