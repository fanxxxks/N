# P59 exact-reference reconstruction, before any account run

The explicit backup recovery at `a779ac36ed60bf7c75306ba2952309b1db0bc490`
was refused before loading or account execution: historical daily_price_limit
partition fingerprints differ. The initial metadata probe that saw a 2026
extension did not establish equality for all historical partitions. Preserve
the failed attempt in `reports/task2_20261007_recovered/ledger.jsonl`. No
income result or altered trading constraint is accepted from this attempt.

One bounded replacement preparation is authorized by the user's task-two
implementation instruction: reconstruct the original reference in a new,
disposable private copy, never restore/modify the production or backup files.
This exception applies only to the preparation described below, superseding
the previous addendum's no-restoration statement for this private copy.

The existing published generation 7 vs backup generation 5 primary manifests
have identical daily_bar, fundamental_pit and stocks fingerprints (and schema).
Only primary sw_industry_index differs. The actual current primary drift was
exactly these three other tables. Therefore copy the current primary file and
its WAL to a new task-owned directory, require source file hashes to remain
stable during copying, replay/checkpoint only the private copy, and replace
only its three drifted tables with the existing backup's rows using its exact
schema columns. Keep the original generation 7 publication rows and all other
tables; do not mint or publish another snapshot.

Use the original current v2 file read-only, including its original generation
7 historical price-limit corrections. Require full uncached confirmation of
both databases to the original composite identity
`d36024cd1b197a8edcf976b2d950b3506bb0fae022f121d6aef06f2a281cde55`.
If reconstructed primary or current v2 differs at all, refuse. No fingerprint
is manufactured, no mismatched data is relabeled, and no cache key is rewritten.
The existing factor4 cache must hit that same confirmed identity and all three
frozen full matrices must match. All original signal/config/risk/cost rules hold.

Add only an explicit `--primary-snapshot` path to the measurement CLI, mutually
exclusive with the refused backup option. Preserve the real primary/v2 paths
and copy/restore procedure and hashes in provenance. This is a one-off local
input preparation, not a production data repair/migration framework. Source
databases, WAL and all old evidence stay untouched. Six account runs are still
the maximum; zero have run. No further input-reconstruction route or research
scope expansion is allowed if this attempt cannot confirm the exact reference.
