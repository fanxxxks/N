# Phase 4 measurement log (T4-01 experiment ledger + stitched OOS protocol)

Every PR records its invariants and measurements here.  Test counts are
`pytest tests` from the repo root, excluding `tests/test_webapi.py` (a
pre-existing environment blocker: starlette 1.3.1's TestClient requires
`httpx2`, which is not installable in this offline environment).

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, post-Phase-3) | b71415e | 818 | — | Phase-3 closing count |
| T4-01 experiment ledger + stitched OOS protocol | (feat commit) + merge | 887 | +69 | 0 regressions; PROTOCOL_VERSION 19 → 20; +17 ledger / +14 regime / +18 stitched / +20 promotion tests |

## What changed (Phase-4 rules)

1. **2021–2026 history is development/validation data.**  It has been
   viewed repeatedly, so it is never called a "final holdout" again.
   The data regime is declared in `data/holdout_registry.json`
   (`ashare_model.regime`): everything ≤ `dev_cutoff` is dev; the next
   true final evaluation may only use future data or a strictly locked
   slice (locked *before* anyone views it, bound to a `dataset_id`).
   Protocol runs refuse any fold that reaches a locked slice before the
   first trial.
2. **Nested walk-forward.**  The inner loop tunes formula / reward /
   hyperparameters inside each fold's training window (unchanged,
   t+2 time contract); the outer loop performs exactly one algorithm
   evaluation per (fold, seed).  Adjudication now consumes the
   **stitched** outer OOS returns: one trial = one (candidate, seed)
   concatenated series, and Sharpe / Deflated Sharpe / max-t / top-trial
   are all computed on that stitched matrix (`stitch_oos_series`,
   `stitched` artifact block).
3. **Automatic trial ledger.**  Every trial opens as `running` before
   any work and closes as `succeeded`/`failed` on both paths
   (`ashare_model.ledger`, append-only JSONL with a sha256 hash chain);
   a crash is recorded, never silently dropped; a run finishing with
   open trials is `ledger.tainted`, never presented as clean.
4. **Champion/Challenger promotion** (`ashare_model.promotion`) requires
   all five gates at once: data & formula P0, statistical significance,
   excess-return & risk constraints, cost/capacity stress (0.5x/1x/2x
   costs × 0.1x/1x/10x capital), and ≥ 1 complete future paper-trading
   observation window (`data/paper_windows.json`).

## T4-01 invariants (asserted by tests)

1. **Stitching is chronological and complete**: fold OOS series are
   concatenated in fold order; failed folds are recorded in
   `failed_folds` and contribute no returns; a (candidate, seed) with no
   succeeded fold produces no trial; two succeeded rows for the same
   (candidate, seed, fold) raise `ValueError` (`tests/test_stitched_oos.py`).
2. **One trial = one stitched series**: DSR / max-t / top-trial count
   (candidate, seed) pairs, not fold rows — 3 candidates × 2 folds is a
   3-trial matrix (`n_trials == 3`, `t_best == 100`); extra trials from
   prior artifacts are stitched *separately* (a prior run's trial is
   never merged into this run's series).
3. **The ledger never loses a trial**: a trial context manager closes
   `succeeded` on success and `failed` with the exception text on any
   crash; a trial that never closes stays visible in
   `open_trials()`/`finalize()`; reload verifies sequence continuity and
   the hash chain and raises `LedgerIntegrityError` on any gap, tamper
   or truncation; closing is one-shot.
4. **Locked data is untouchable**: a fold reaching a locked slice is
   rejected before the first trial (train windows start at the beginning
   of history, so a fold that reaches the lock consumes locked data); a
   lock declared on another `dataset_id` is unverifiable and blocks the
   run; a final evaluation must classify every fold as future or locked.
5. **Promotion is all-gates**: each of the five gates returns
   `{passed, reasons}`; a single breached gate (old protocol version,
   dataset mismatch, dev-data final evaluation, ineligible formula,
   DSR < 0.95, max-t p > 0.05, drawdown > 0.30, stress-cell breach,
   missing/short/unfinished/dev paper window) flips `promoted` to False
   with an auditable reason.
6. **The stress grid reproduces the artifact**: the 1x1
   (cost × capital) cell re-scores the champion through the identical
   engine path and matches the plain evaluation to 1e-9
   (`test_stress_champion_grid_on_synthetic_data`).
7. **CLI wiring**: protocol and selfcheck runs write the ledger and
   regime provenance into the artifact (`ledger` block with
   `path`/`run_id`/`tainted`/`n_trials`, `data_regime` block with per-fold
   window classifications); CLI smoke tests pin the v20 schema.

## Research-validity evidence (not pytest)

* The stitched matrix removes the per-fold averaging bias: a candidate
  that is good in one fold but bad in others is now measured on its
  *full* OOS path, so its Sharpe / DSR / max-t reflect the path a real
  deployment would have delivered, not the mean of per-fold snapshots.
* The ledger's hash chain + open-trial taint make "we forgot to record
  a failed run" structurally impossible: a run either closes every
  trial it opened or is flagged tainted.
* The regime registry turns the "never view the holdout" rule into an
  enforced gate: promotion is refused on dev data, and a locked slice
  declared on a different dataset id blocks the run instead of silently
  trusting a stale lock.
* The self-check (pure-noise candidates) still reports insignificant
  DS/max-t through the stitched path (`test_selfcheck_rows_stitch_into_one_trial`).

## Version bumps (semantic changes)

| Module | Before | After |
|---|---|---|
| ashare_model.evaluation.PROTOCOL_VERSION | 19 | 20 (stitched trial matrix; per-fold row matrix superseded) |
| ashare_model.ledger | — (new) | ExperimentLedger / LedgerEntry / LedgerIntegrityError |
| ashare_model.regime | — (new) | DataRegime / LockedSlice / RegimeRegistry / HoldoutViolation |
| ashare_model.promotion | — (new) | PromotionThresholds / PaperWindowRegistry / evaluate_challenger / stress_champion |

## Migration / rejection policies

* **Prior protocol artifacts (v19 and older) stay readable and usable
  as dev/validation evidence**: their raw rows load with `--trials` and
  stitch under the same rule as v20 rows (a prior run's trial is a
  separate trial, never merged into the current run's series).  They are
  *not* eligible for Champion/Challenger promotion: G1 rejects any
  artifact whose `protocol_version` is not v20 and any artifact without
  a `dataset_id` (pre-T1-01 legacy).
* **The ledger is append-only and never migrated**: entries written by
  earlier runs stay in place; new runs append with a new `run_id`.
  A ledger whose hash chain fails verification is *rejected*, not
  repaired — integrity errors surface instead of being papered over.
* **No persisted execution artifacts change**: the backtest engine, the
  golden execution spec, the optimizer and the matcher are untouched;
  the full-suite regression count is zero.
* **`aggregates` stays row-level** (per-fold × seed drill-down, as in
  v19); the v20 adjudication block is `stitched` (per (candidate, seed)
  series).  The two granularities are documented in the artifact schema
  and are not interchangeable.
