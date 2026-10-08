# P59 task-two actual measurement

Local branch `codex/task2-continuous-account`, task-one base
`1ad7686a70cb387c940b0c5053acf8339df035a5`. Preregistration `45db5eb`,
input-recovery preregistration `2630ebf`, exact-reference reconstruction
preregistration `e319d4b`. Measured implementation:
`dd5453a92d3613c6b9161000b176259d7faf39fc`. Subsequent commits only add evidence.
Original `D:/minequant/AlphaGPT` code, configuration, data, WAL, backups and old
WP18 evidence were never edited. No push, PR, merge or strategy promotion.

## Actual commands and environment

Working directory: `D:/minequant/AlphaGPT-task1`; interpreter:
`D:/minequant/.venv/Scripts/python.exe`. Windows 11, Python 3.13.12,
torch 2.11.0+cu128, CUDA runtime 12.8; NumPy version and full version-owner
matrix in report provenance. Accounts use the existing CPU NumPy engine and
cached factor tensors; no search seed, proposals or model training.

```
python scripts/task2_continuous_account.py --source-root D:/minequant/AlphaGPT --primary-snapshot data/task2_reference_20261007_clean/ashare.duckdb --output reports/task2_20261007_exact_reference
python -m pytest -q tests/test_p59_continuous_account.py tests/test_p58_small_capital_execution.py tests/test_backtest.py tests/test_p35_share_accounting.py tests/test_golden_parity.py tests/test_b04_load_cache_performance.py tests/test_artifact_versions.py tests/test_ci_sharding.py tests/test_time_contract.py
python scripts/check_test_shards.py --check
python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi
git diff --check
```

On exact implementation: **165 passed in 100.57 s, zero warnings/skips**;
raw output `reports/task2_20261007_exact_reference/exact_tests.txt`.
125 test files fully assigned across four shards; compileall and diff checks
passed. No existing tests were modified, no tolerance/skip was weakened.
New tests were registered. Earlier RED and the snapshot-key correction are
recorded in `p59_continuous_account_implementation.md`; final selector RED was
one failure (absent CLI option), then all 20 focused cases passed.

Full Python suite, CI warning-merge job, Web jobs, alternative Python/runtime
matrix and parallel-vs-serial suite were **not run**: this local task is scoped
to accounts/execution/input guards and does not include push/PR/merge. This is
not a push-ready or complete CI claim.

## Input failures, recovery and provenance

1. `cf1b0a2` original current primary load refused content drift in daily_bar,
   fundamental_pit and stocks against published generation 7. Preserved in
   `reports/task2_20261007/ledger.jsonl`; no account ran, no factor rebuilt.
   That implementation's exact adjacent test run: 159 passed in 119.00 s.
2. Read-only backup metadata probes initially identified a 2026 price-limit
   extension, but that did not establish historical equality. `a779ac3`
   explicitly tested the paired whole backup and **refused historical
   daily_price_limit mismatch**. Preserved in
   `reports/task2_20261007_recovered/ledger.jsonl`; again zero account runs.
3. Original and backup published primary fingerprints matched exactly for the
   three drifted tables. Current primary file and WAL were copied only to
   new task-owned paths with before/after file SHA equality; original v2 was
   kept read only. First private DELETE/INSERT preparation failed at 1 GB
   commit memory, then on unique ART index commit. These were discarded local
   preparation failures, not account variants. That private directory remains
   unused; no failed result was replaced with a successful return curve.
4. A fresh private copy used CTAS for only those three tables, preserving
   column types and rows (indices are not read by the account). Counts:
   daily_bar=4,919,116, fundamental_pit=215,951, stocks=5,548. No publication
   rows, other tables, original/backup files or factor cache entries were
   rewritten. Copy preparation took 20.54 s; source hashes stayed stable.
   Preparation and SHA record is embedded in successful provenance.
5. Full uncached existing-authority confirmation of the private primary and
   original v2 **matched the exact original composite**
   `d36024cd1b197a8edcf976b2d950b3506bb0fae022f121d6aef06f2a281cde55`.
   Existing factor4 cache `4ab7587af59d…` hit; the three entire frozen matrices
   matched every preregistered SHA256. No identity bypass or cache reseeding.
   All 2021–2025 B directions, 10-day labels and causal cutoffs were checked.

Some diagnostic probes triggered a Windows pyarrow import access violation
(outside engine/test runs); this was recorded as an environment probe failure,
not suppressed as a healthy data confirmation. No dependency/runtime changes
were introduced. The successful CLI and exact test run completed normally.

## Budget, outputs and actual findings

Exactly **six** normal-fee accounts: three fixed capitals × active/passive;
all six called the existing engine once for all years. Two refused input
attempts produced zero account trials. No frequency, weight, holding-count,
threshold or fee search. The successful load plus six accounts took **308.07 s**;
the entire input-recovery/account session remained within the 90-minute bound
(first input start 2026-10-08 02:18:47 UTC, last account 03:09:19 UTC).
Observed successful-process working-set peak during sampling was at least
2,758 MB; this is a sampled lower bound, not a complete peak-memory claim.

Loaded axis 20140102–20260916, 3,091 dates × 1,630 stocks. Continuous marked
account axis **20210105–20251231**, 1,210 daily return periods; first fills
20210108. Cadence generated once on the full date axis. All annual wealth
chains, CSV fee components, summary terminal assets, cash nonnegativity,
active/passive first fills and passive absence of later orders were separately
reconciled by reading actual output CSVs. Reopening the existing ledger verified
its hash chain and exactly six closed account trials. Audit in `account_audit.json`.
No degraded-price source codes or missing terminal held prices were observed.

Absolute active net profits: 30k=57.80 RMB, 50k=1,260.47 RMB,
100k=10,755.76 RMB. Their matched passive profits were 6,415.25 / 17,064.26 /
38,991.59 RMB. All active compounded excesses versus that passive basket were
negative. Mean active invested fractions were 8.69% / 14.70% / 16.22%; mean
actual holding counts 2.8 / 5.0 / 5.7. After first funding, under-20 days were
100% / 97.51% / 98.34%, so these fixed candidates fail the original continuous
dispersion requirement. 10k remains structurally inapplicable. Passive risk and
concentration drift are disclosed, not presented as an investment recommendation.

Actual fees: 950.70 / 3,085.49 / 4,312.75 RMB, with four separate components.
Planned vs filled turnovers/orders differ substantially and are explicitly
labeled. No positive-net-wealth result is relabeled as proof of active alpha.
The old adjusted-return reinvestment approximation, fixed historical fees and
non-liquidated terminal valuation are explicit model limitations.

Decision and full table in `reports/task2_20261007_exact_reference/DECISION.md`
and `RESULTS.md`; six JSON/summary rows, full daily/fill CSVs and source/config/
direction/signal/version hashes retained. Failed inputs retained alongside.
Task two delivered; tasks three/four and all deferred directions were not run.

## Terminal-field review correction

Final review found v1's terminal exposure was the last entry observation,
not the terminal exit mark. Independent preregistration `c46c96f` preceded
RED: 99,000 invested wealth doubles to 198,000, cash remains 849.51;
terminal NAV=198,849.51, so terminal exposure differs from prior entry exposure.
The v1 assertion failed; reporting-only implementation
`b2f1afbe00fc2737e6f5c76b95426d79acd53d6a` fixes it and bumps the local summary
owner to version 2. Focused 21 cases passed in 16.92 s before this commit.

`results_corrected.json` and `summary_corrected.csv` were newly written from
the six previously audited summaries using that same public exposure helper;
zero engine reruns. Only local version/terminal-exposure/last-entry fields
change. Exact original v1 results bytes remain untouched; all existing amounts,
returns, drawdowns, costs, average exposures and conclusions match exactly.
Provenance carries original engine SHA, summary-correction SHA, original JSON
SHA256 and zero-rerun count. `terminal_observation_correction.json` gives every
old/new exposure. Raw observations continue to refer to their entry_date.

On exact reporting-correction commit: **166 passed in 83.11 s**, zero warnings/
skips; raw output `exact_summary_v2_tests.txt`. The same adjacent command, shard
union, compileall and diff checks passed. No full-suite/runtime-matrix/CI claim
is added by this focused rerun. Both tested SHAs and outputs remain retained.
