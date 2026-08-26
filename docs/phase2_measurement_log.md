# Phase 2 measurement log (T2-01 .. T2-03)

Every PR records its invariants and measurements here.  Test counts are
`pytest tests` from the repo root, excluding `tests/test_webapi.py` (a
pre-existing environment blocker: starlette 1.3.1's TestClient requires
`httpx2`, which is not installable in this offline environment).

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, pre-Phase-2) | 25bc994 | 672 | — | 20:39; webapi excluded; matches Phase-1 closing count |
| T2-01 AST canonicalization + semantic cache | 4d81bb5 + merge 737fc29 | 726 | +54 | 0 regressions; 16:43; PROTOCOL_VERSION 17→18 |
| T2-02 DEAP GP + Optuna TPE baselines | acf8216 + merge 5952217 | 737 | +11 | 0 regressions; 17:57; deap==1.4.4 / optuna==4.9.0 added to optional deps |
| MTPHead removal | <commit> + merge | <after> | +<n> | MODEL_VERSION 1→2; old .pt checkpoints rejected (no multi-task supervision existed) |

## T2-01 invariants (asserted by tests)

1. **Budget unit = unique semantic formula evaluation.**  The semantic
   cache (`ashare_model/semantic_cache.py`) bills one unit only when a
   formula's semantic class — canonical AST hash, or calibration
   fingerprint + complexity bill — was never evaluated before.  Structural
   duplicates (commuted ADD/MUL), degenerate constant-producing formulas
   (`SUB(x,x)`, `DIV(x,x)`, `CORR*(x,x)`, `ADD(x,NEG(x))`, propagated
   through enclosing operators) and numerically equivalent formulas
   (`ADD(x,x)` vs `x` under the same bill) never consume budget.
2. **Cache key carries the full evaluation context**: canonical AST hash,
   `dataset_id`, `reward_version`, `protocol_version`, `window_id` — a
   score is never reused across datasets or measurement generations
   (`test_key_changes_with_every_context_field`).
3. **Fingerprint is a fixed rule**: the head of the training window (all
   stocks, first 60 columns) — identical for every searcher, so
   RL-vs-baseline budget comparisons are fair by construction.
4. **Degenerate formulas are rejected pre-evaluation** (never executed,
   never billed); the trainer's operator-coverage gate still hard-fails
   bare-factor runs.
5. **Complexity bills stay exact**: the semantic class includes the AST
   complexity bill, so numerically equivalent formulas with different
   penalties are distinct classes (`test_semantic_class_includes_
   complexity_bill`).
6. **Feature registry is deterministic and versioned**: PIT tier,
   coverage, correlation cluster (|corr| ≥ 0.9 on the calibration slice)
   and deprecation status per vocabulary feature
   (`tests/test_feature_registry.py`).

## Research-validity evidence (not pytest)

* The semantic budget is a measurement change, not a claim: T2-03 runs the
  RL admission experiment (≥ 5 independent init seeds, identical unique-
  semantic-evaluation budget vs random/GP/TPE, best-so-far area + OOS
  active IR) and records the decision here.  T2-01 only establishes the
  ledger every searcher is billed on.

## T2-02 invariants (asserted by tests)

1. **Mature frameworks, not hand-rolled engines**: strongly-typed GP uses
   DEAP's official GP module (typed primitive set, half-and-half
   generation, one-point crossover, uniform mutation, tournament
   selection); TPE uses Optuna's ``TPESampler``.  The repo code only maps
   trees/tokens to the formula grammar and binds evaluation to the shared
   budget (`tests/test_gp_search.py`, `tests/test_tpe_search.py`).
2. **One budget ledger for every searcher**: `SemanticBudgetEvaluator`
   (shared by random / GP / TPE) bills unique semantic formula evaluations
   only; the GP and the random baseline never double-bill the same class.
3. **Search space parity**: GP trees are capped so every proposal fits the
   policy's ``max_formula_len``; TPE proposes through the same
   ``build_action_mask`` legality rules the policy samples under, so no
   trial is wasted on invalid sequences.
4. **Determinism**: both baselines are deterministic in ``seed``
   (same seed -> identical best-so-far curve and selection).
5. **Best-so-far curves**: both baselines record ``(cumulative budget,
   best validation reward)`` with a monotone curve, for the T2-03
   admission-area comparison.
6. The protocol's own random-search row keeps its inline v18 loop; it is
   moved onto the shared evaluator together with the GP/TPE protocol rows
   in T2-03 (one wiring PR, one bump).

## T2-02 dependencies

* `requirements-optional.in` gains `deap` and `optuna` (exact pins
  `deap==1.4.4`, `optuna==4.9.0` in `requirements-optional.txt`);
  `requirements.lock` re-frozen (adds alembic/colorlog/greenlet/Mako/
  moocore/platformdirs/SQLAlchemy as Optuna/DEAP transitive deps).
* The optional-deps file is the Phase-2 research tier; `requirements.txt`
  (the production install) is unchanged — the search baselines are
  research tooling, not production dependencies.

## Version bumps (semantic changes)

None in T2-02: the protocol's candidate pool is unchanged (the GP/TPE
rows join the protocol in T2-03, which bumps PROTOCOL_VERSION).

## Migration / rejection policies

* No persisted artifacts change in T2-02; the evaluator and the search
  modules are additive.

## MTPHead removal

* The multi-task head had no multi-task supervision anywhere in the
  training pipeline and the trainer discarded its router probabilities
  (``logits, value, _ = model(inp)``) — dead complexity kept alive only
  by its name.  Removed: the policy is a single linear head + critic.
* **Version bump**: ``MODEL_VERSION`` 1 → 2 (new constant in
  ``ashare_model/alphagpt.py``, recorded in training artifacts).
* **Migration / rejection**: ``data/ashare_model.pt`` checkpoints from
  v1 carry ``mtp_head.*`` state keys and are **rejected** (no loader
  exists for them; nothing in the repo restores checkpoints into the
  model — the formal artifact is ``best_ashare_strategy.json``, which is
  unaffected).  Re-training is the only migration path, matching the
  "no old-artifact compatibility" policy for architecture changes.

## Version bumps (semantic changes)

| Module | Before | After |
|---|---|---|
| ashare_model.evaluation.PROTOCOL_VERSION | 17 | 18 |
| ashare_model.semantic_cache.SEMANTIC_CACHE_VERSION | — (new) | 1 |
| ashare_model.feature_registry.FEATURE_REGISTRY_VERSION | — (new) | 1 |

Training artifacts gain (additive): `protocol_version`,
`semantic_cache_version`, `unique_semantic_evals`, `semantic_cache_stats`;
history rows gain `unique_semantic_evals` / `semantic_dedup_frac`.

## Migration / rejection policies

* Pre-v18 artifacts keep loading: the new fields are additive and
  protocol/reward versions are recorded per artifact; a v17 artifact is
  never silently compared against v18 measurements.
* The semantic cache is runtime-only (no persisted artifacts): no
  migration needed; the LRU is bounded by the trainer's `_REWARD_CACHE_CAP`.
