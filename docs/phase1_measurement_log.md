# Phase 1 measurement log (T1-01 .. T1-05)

Every PR records its invariants and measurements here.  Test counts are
`pytest tests` from the repo root, excluding `tests/test_webapi.py` (a
pre-existing environment blocker: starlette 1.3.1's TestClient requires
`httpx2`, which is not installable in this offline environment).

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, pre-Phase-1) | 5840943 | 581 | — | 311 s; webapi excluded |
| T1-01 dataset manifest + dependency lock | 9d27769 + merge b5934d5 | 607 | +26 | 0 regressions; manifest created_at flake fixed later (T1-02) |
| T1-02 no-signal semantics + permutation invariance | 14925fe + merge e370e3c | 622 | +15 | 0 regressions; 14 new tests + 2 sim tests, 4 updated to the new contract |
| T1-03 robust IC/effective-n/stability gates + complexity billing | 3312f11 + merge 603ceab | 647 | +25 | 0 regressions; 23 new tests, 9 updated to v12 semantics |
| T1-04 universe alignment + portfolio objectives + Pareto | 5482d1d + merge adab99e | 659 | +12 | 0 regressions; 12 new tests, 6 updated to v13 semantics |
| T1-05 matched-budget harness + completion gates | b4818d9 + merge 0156000 | 672 | +13 | 0 regressions; 13 new tests; weight quantization fix (see below) |

## Completion gates (all asserted by tests on the production default config)

1. **All-zero / two-valid-day / extremely-sparse signals rejected** —
   `tests/test_completion_gates.py::test_gate_all_zero_signal_rejected`,
   `test_gate_two_valid_day_signal_rejected`,
   `test_gate_extremely_sparse_signal_rejected` (default RewardConfig).
2. **Stock-order permutation never changes any measurement** —
   `test_gate_permutation_invariance_full_chain` (scorer, exact to 1e-9)
   plus the T1-02 engine/basket/scorer permutation tests.  The gate
   exposed a real bug: ~1e-16 float summation noise in the
   scale-to-budget weight construction fabricated phantom micro-orders
   that paid the full 5-yuan minimum commission, flipping order counts
   between stock permutations.  Weights are quantized to 1e-12 in the
   engine and the reward basket (T1-05 commit).
3. **Reward stably positively correlated with OOS active IR** —
   `tests/test_baseline_harness.py::test_completion_gate_reward_correlates_with_oos_active_ir`:
   rho > 0.4, permutation p < 0.05, deterministic seed; plus the pure
   `reward_oos_correlation` unit tests.  The harness measures the v13
   reward (annualized active IR of the gross basket vs the equal-weight
   benchmark, minus cost and complexity) against the engine's OOS excess
   series.
4. **single_weight_cap never broken** —
   `test_completion_gate_single_weight_cap_never_broken`: an adversarial
   sweep over cap regimes (0.03 / 0.1 / 0.5 / 1.0), top_n (2 / 5 / 20),
   under-fills, force-holds and degenerate days asserts every engine
   position weight <= cap and the book sum <= 1.

## Research-validity evidence (not pytest)

* The correlation harness is a measurement, not a test: run
  `python scripts/baseline_harness.py --config config/ashare_config.yaml
  --fold 0` for the live-dataset reward<->OOS active IR report.  The
  synthetic gate uses the same scorer/engine path with the same
  weights, costs and masks.
* The matched-budget baseline (T1-05) gives random search the trained
  candidate's exact unique-evaluation budget per fold
  (`tier.steps x tier.batch_size`), recorded as
  `random_budget_matched` / `random_budget` in protocol artifacts
  (PROTOCOL_VERSION 17).

## Version bumps (semantic changes)

| Module | Before | After |
|---|---|---|
| ashare_model.reward.REWARD_VERSION | 10 | 13 |
| ashare_model.evaluation.PROTOCOL_VERSION | 12 | 17 |
| ashare_data.manifest.MANIFEST_VERSION | — (new) | 1 |
| CandidateScore artifact schema | 12 fields | 17 fields (complexity_cost, active_ir, risk_exposure, average_turnover, capacity_utilization), legacy payloads fall back |

## Migration / rejection policies

* Pre-T1-01 artifacts: no `dataset_id` — accepted as legacy, recorded as
  `null`; artifacts whose `dataset_id` differs from the current database
  are rejected (`DatasetIdMismatch`) when loaded as trial rows.
* Pre-v13 CandidateScore payloads: `from_payload` falls back to the old
  field names; new fields default to 0/None.
* Rewards from different REWARD_VERSION generations are never compared
  silently (recorded per artifact, enforced by the version fields).
* `requirements.txt` switched from lower bounds to exact pins;
  `requirements.in` holds the spec; `requirements.lock` is the full
  frozen environment; `scripts/freeze_lock.py --check` gates drift.
* MLflow tracking is additive and opt-in (`MLFLOW_TRACKING_URI`); the
  archive JSON stays the primary record; every failure mode is a
  structured, non-fatal outcome.

## Pre-existing environment blockers (unchanged by Phase 1)

* `tests/test_webapi.py` cannot collect on this machine (starlette 1.3.1
  requires `httpx2`, not installable offline).  CI (ubuntu, python 3.12)
  is unaffected in principle but has not been exercised from here.
