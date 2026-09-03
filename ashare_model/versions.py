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
"""

PROTOCOL_VERSION = "25"
