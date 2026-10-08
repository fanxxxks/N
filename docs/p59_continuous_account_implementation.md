# P59 implementation notes

Contract: `p59_continuous_account.md`, independently committed as `45db5eb`.
Base: task-one `1ad7686a70cb387c940b0c5053acf8339df035a5`.
Local branch: `codex/task2-continuous-account`; original user checkout untouched.

## Changes and compatibility

- The existing backtest records each filled trade and its already-computed cost
  components in `BacktestResult.trades`. It adds per-period account observations,
  execution rejections and market block counts to `construction_diagnostics`.
  Historical `positions` keys stay unchanged for existing dashboard consumers.
- `AshareDataLoader(require_factor_cache_hit=True)` refuses a missing/rejected/
  disabled cache before expensive factor rebuilding or any cache write. Default
  callers keep their previous behavior. The P59 CLI is the real consumer.
- `continuous_account.py` validates and concatenates frozen yearly B directions,
  calls the existing equal-weight rank combiner, derives annual returns from
  continuous wealth, and reconciles filled costs/cash/daily wealth for summaries.
- `scripts/task2_continuous_account.py` loads confirmed inputs once and calls the
  existing engine once per account, for exactly three capitals and two holding
  policies. It exports the new local summary format (version 1), full daily/fill
  CSVs, provenance and the existing ledger's success/failure hash chain. These
  development reports are not strategy/promotion/resume boundary artifacts.
- The CLI consumes literal frozen YAML plus existing config factories; it does
  not merge mutable UI runtime overrides or environment path overrides. Effective
  configurations and YAML hashes are recorded. Inputs are read only; an existing
  output directory is refused, preserving all previous evidence.
- Execution 6, constructor 2, protocol 29, reward 15 and artifact schema 3 do not
  change: existing trading, returns and consumer snapshots retain their semantics.
  No new dependencies, production defaults, frontend, ETF data or research search.

## RED and precommit verification

New `test_p59_continuous_account.py` assertions derive from the preregistered
requirements and independent hand cash/wealth ledgers. Initially the new module
was absent; after adding the reporting helper but before engine/loader wiring,
the same test file had **5 failed, 8 passed**: no fills/cost facts, absent account
observations and no cache-only refusal. No existing tests were edited.

The first adjacent run had **1 failed, 157 passed**: the existing backtest test
requires the exact historical snapshot key set. The implementation was corrected
to put observations in diagnostics, rather than weakening that assertion.
Then the adjacent set passed **158 tests**, and the final focused set passed
**14 tests** after adding the hand-calculated drawdown/year oracle.

An independent precommit replay compared the committed task-one engine with the
new engine over 16 cases: capitals 10k/30k/50k/100k, minimum 1k/5k, and priority/
proportional. All pre-existing equities, daily returns, plan turnovers, snapshot
records, benchmark curves, metrics, cost fractions, target/buy/sell arrays, cash,
shares and prior diagnostic fields matched exactly. This is software parity,
not evidence of profitability. Compileall and the 125-file shard union check
passed. Exact-commit verification and historical measurements are recorded in
the subsequent measurement log, after they actually run.

Rollback is reverting this implementation commit and retaining its immutable
report directory. No source data or historical WP18 report migration is required.
