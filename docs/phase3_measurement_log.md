# Phase 3 measurement log (T3-01 .. T3-02)

Every PR records its invariants and measurements here.  Test counts are
`pytest tests` from the repo root, excluding `tests/test_webapi.py` (a
pre-existing environment blocker: starlette 1.3.1's TestClient requires
`httpx2`, which is not installable in this offline environment).

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, post-Phase-2) | 9fe4b06 | 749 | — | Phase-2 closing count |
| T3-01 constrained portfolio optimizer | 7a80241 + merge 88d0793 | 791 | +42 | 0 regressions; cvxpy==1.6.5 added to the production spec + frozen locks |
| T3-02 golden parity (unified execution spec) | e9f8afc + merge b71415e | 818 | +27 | 0 regressions; EXECUTION_SPEC_VERSION 1 |

## T3-01 invariants (asserted by tests)

1. **The QP core is CVXPY/OSQP** — never a hand-written solver.  The
   objective is `alpha'w - risk_aversion * w'Sigma w -
   turnover_cost * ||w - w0||_1 - impact_cost * sum_i ((Δw_i *
   capital / adv_i)^2)`; every convex constraint is exact inside the QP:
   long-only, `sum(w) <= 1 - min_cash`, single-name cap, per-industry
   cap, Beta/size exposure ranges, per-name ADV participation
   (`|Δw_i| * capital <= rate * adv_i`) and the L1 turnover budget
   (`tests/test_portfolio_optimizer.py`).
2. **Infeasible problems raise `PortfolioOptimizationError`**; an empty
   universe (no finite alpha) returns an all-zero book with status
   `"empty"` without calling the solver; non-finite alpha entries are
   forced to zero weight (and their coefficients zeroed before the
   objective so they can never poison the solve).
3. **Fail-fast data validation**: every configured constraint requires
   its data (`industry_cap` -> industries, `beta_range` -> beta,
   `adv_participation`/`impact_cost` -> adv, `risk_aversion > 0` -> cov);
   shapes and ranges are validated; a constraint is never silently
   dropped.
4. **The two non-convex constraints are a documented shrinking
   projection** after the QP: `min_trade_amount` suppresses trades below
   the yuan threshold (a held name whose reduction is sub-minimum keeps
   its previous weight), `max_positions` is a hard ceiling on
   `count(w > 0)` (forced liquidation sells are exempt from the
   min-trade filter), and weights only shrink — never renormalized
   upward (the repo-wide T1-02 contract) — so caps, industry, budget,
   ADV and turnover stay satisfied; Beta/size exposure after projection
   is reported in `diagnostics` as the residual, never silently hidden.
5. **Solver-noise cleanup**: OSQP's default 1e-5 tolerance can return
   ~1e-6 phantom weights at true-zero entries; entries below 1e-5 are
   zeroed so no phantom micro-order ever pays the minimum commission in
   a downstream execution path.
6. **Determinism**: identical inputs produce bitwise-identical weights.

## T3-02 invariants (asserted by tests)

1. **Same signals, three paths, one contract**: the golden harness
   (`ashare_portfolio/golden.py`, `EXECUTION_SPEC_VERSION = 1`) runs the
   vectorized engine, the matcher in lot-free continuous-share mode and
   the matcher in whole-lot mode; `verify()` raises
   `GoldenParityViolation` on any breach.
2. **Lot-free mode is exactly identical to the engine** (1e-9 relative,
   `tests/test_golden_parity.py::test_lot_free_matches_engine_exactly`).
   The engine's convention is a frictionless-rebalanced book: positions
   re-marked at target weights each entry, only weight deltas traded,
   fees on weight-delta notionals.  The lot-free matcher reproduces it
   through the authoritative `SimBroker` (blocking, T+1, prices, minimum
   commission), so any disagreement between the matcher's rules and the
   engine's surfaces as a divergence.
3. **Lot-mode differences come only from recorded integer lots and cash
   residuals**: the per-day residual `(Δlot - Δfree)` decomposes exactly
   into the recorded `rounding` PnL (integer-lot quantity differences),
   `fee_diff` (real trade fees minus weight-delta fees) and `carry`
   (sub-lot and affordability cash); the bookkeeping identity
   `sum(residual) == lot_equity(T) - free_equity(T)` holds exactly
   (`test_lot_residual_decomposes_into_recorded_rounding_and_cash`).
4. **Whole-lot scenarios are exact**: with constant prices every target
   notional is a whole lot and lot mode == engine with zero residual
   (`test_whole_lot_scenario_is_exact_in_lot_mode`).
5. **The spec's blocking rule applies to any weight source**:
   buy-blocked / outside-universe names are never freshly bought;
   sell-blocked reductions are force-held at the previous weight; never
   renormalized upward.  For engine weights it is a no-op; for optimizer
   weights it keeps both paths consistent (`apply_blocking_rule`).
6. **Coverage scenarios** (each a golden test): suspension (zero
   volume), limit-up buy block, limit-down exit force-hold, T+1 flow
   (buys fill with zero available quantity; exits sell the next
   execution day), delisting (bars end: force-held, liquidation attempts
   skipped with `missing_bar`/`suspended`, never filled), qfq-adjusted
   price series (2:1 split-like jump), fees in both paths, insufficient
   cash (lot mode: affordability-capped, cash never negative), missing
   bars (non-finite OHLCV = missing bar, same blocking as the engine's
   zero-cell rule).
7. **Stress grid keeps the contract exact**: cost/slippage/stamp
   multipliers 0.5x / 1x / 2x; one-day delayed execution
   (`execution_delay=2`, an additive engine parameter with default 1 =
   unchanged behavior); missing bars; alternating one-word ±10% limit
   moves; capital scales 1e5 / 1e7 / 1e9; lot sizes 100 / 1000.
8. **Capstone (factor -> optimizer -> matcher)**: T3-01 optimizer target
   weights executed through the spec keep the bookkeeping identity, and
   the executed book respects the optimizer's single-name / industry
   caps and position ceiling within lot tolerance
   (`test_optimizer_weights_execute_with_constraints`).

## Research-validity evidence (not pytest)

* The parity contract is a cross-implementation check: the free path and
  the lot-free matcher path compute the same equity through independent
  code (the engine's internal loop vs the broker-driven harness), so a
  divergence means one of the two execution models disagrees with the
  other — the golden test turns that into a hard failure instead of a
  silent drift.
* The residual decomposition is measured and recorded per day (not
  asserted away): the harness reports `rounding`, `fee_diff`, `carry`
  and `residual` for every signal period, so lot-mode cost attribution
  is auditable per rebalance.
* The optimizer's correctness is asserted analytically where possible:
  the impact test pins the closed-form optimum (alpha=[0.05,0.05],
  adv=[1e6,1e6], capital=1e6, impact_cost=1 -> w_i = 0.025), and the
  turnover test pins the hold-the-book optimum under a cost threshold.

## Version bumps (semantic changes)

| Module | Before | After |
|---|---|---|
| ashare_portfolio.optimizer.PORTFOLIO_OPTIMIZER_VERSION | — (new) | 1 |
| ashare_portfolio.golden.EXECUTION_SPEC_VERSION | — (new) | 1 |
| ashare_model.backtest.AshareBacktestEngine.run | — | +`execution_delay` (default 1 = unchanged behavior; additive) |
| ashare_data.schemas.BacktestResult | — | +`target_weights` (exact per-period vectors; additive, default None) |
| ashare_data.schemas.SimOrder/SimTrade.quantity, ashare_trading.portfolio.PositionState.quantity | int | float (schema widening: lot-free continuous shares; production lot-mode artifacts keep integer values, JSON output unchanged) |
| ashare_trading.matching.SimBroker.execute_orders | — | +`lot_size` (default 100 = unchanged paper-trading behavior; <= 0 = golden lot-free mode) |
| requirements.in / requirements.txt / requirements.lock | — | +cvxpy==1.6.5 (and transitive clarabel/osqp/scs/joblib in the full lock) |

## Migration / rejection policies

* No persisted artifacts change in Phase 3: the optimizer and the golden
  spec are additive modules; the engine/schema/matcher changes are
  additive with defaults that preserve every existing behavior (the
  full-suite regression count is zero at every merge).
* Legacy `BacktestResult` producers (without `target_weights`) keep
  working; the golden harness requires the engine path to record
  `target_weights` and fails fast otherwise.
* The `SimOrder`/`SimTrade`/`PositionState` quantity widening is
  value-compatible: lot-mode runs store integers exactly as before, and
  no loader or consumer coerces the field to `int`.
