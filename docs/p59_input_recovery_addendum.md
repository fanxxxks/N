# P59 input recovery before any account return measurement

2026-10-07, user-authorized task two; no account has run yet.
The first load at implementation `cf1b0a263506dcd78d3c367784efdb36b22d2dc5`
refused: current primary DB no longer matches published generation 7;
daily_bar, fundamental_pit and stocks drifted. Failure retained in
`reports/task2_20261007/ledger.jsonl`; no returns, factor rebuild or source write.

An existing paired backup `data/backup_20260919_p38/{ashare,ashare_v2}.duckdb`
has complete generation 5 with composite identity
`767d955b0fd99507fed95b4275a2bdde5bdab177e1e22c8dc156a45c107e21b2`.
Its existing factor4 cache `fe96090e…` has the same 1630×3091 axis and all three
full factor matrix SHA256 values frozen in P59, including NaN bytes. This is
not conversion of the live database into a historical snapshot.

Before resuming, add one explicit CLI option selecting only this named backup.
No automatic fallback. Both files still must pass full uncached composite
confirmation, cache must hit its own identity, and all frozen factor bytes
must match. The original reference generation 7 identity and the actually
consumed backup identity must both be recorded, never represented as identical.

Additionally compare the two existing published groups' fingerprints through
2025 for primary trade_calendar/constituents/stocks and v2 daily_bar_raw,
daily_adj_factor/daily_price_limit/daily_suspension; reject any difference.
All primary stock rows, calendar and constituency fingerprints must match
exactly (not just current metadata). Backup full content confirmation verifies
that its actual rows match those fingerprints. Differences outside these
consumed inputs are explicitly recorded. Full factor equality covers signal
inputs; equal historical price/constraint/mask sources cover account inputs.

The metadata probe already found the published groups differ only in primary
sw_industry_index, v2 source_ingest_runs and daily_price_limit; the price-limit
extension is in 2026. Current live stocks also gained three 2026 listings and
changed two current ST flags; current ST flags are not used as historical PIT
exclusions. These live rows will not be consumed for the accounts.

This is one input recovery, not an additional strategy implementation/parameter
version or search. Account budget remains six, zero consumed so far. The first
failed input load is preserved; the recovery performs one new confirmed input
load. No production snapshot publication, restoration, DB modification, factor
rebuild or new dataset synchronization is authorized or performed.

New bridge tests must reject a consumed historical table/partition mismatch and
allow unrelated or strictly post-2025 changes. Existing P59 requirements for
causality, fees, cash, continuous accounts, reporting and scope still apply.
