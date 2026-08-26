# Phase 2 measurement log (T2-01 .. T2-03)

Every PR records its invariants and measurements here.  Test counts are
`pytest tests` from the repo root, excluding `tests/test_webapi.py` (a
pre-existing environment blocker: starlette 1.3.1's TestClient requires
`httpx2`, which is not installable in this offline environment).

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, pre-Phase-2) | 25bc994 | 672 | — | 20:39; webapi excluded; matches Phase-1 closing count |
| T2-01 AST canonicalization + semantic cache | <commit> + merge | 726 | +54 | 0 regressions; 16:43; PROTOCOL_VERSION 17→18 |

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
