# COMPATIBILITY — factor_inventory_audit_v2_20260905 (t18)

Evidence directory for the t18 factor retest (77-feature vocabulary on the
Tushare v2 PIT dataset under the D4/D5/D6 semantics).  Per IP-14 01-A6/A7
this directory keeps no executable script: the measurement runs the single
consolidated harness `scripts/factor_inventory_audit.py` (generation
profile `v2`), never a local copy, and the historical evidence scripts of
the 20260831 / v4_20260901 directories stay frozen.

## Frozen provenance

- Tool commit introducing the `v2` profile: `0a8565e` (codex/factor-retest,
  base `db76700` = the captain-ruled D4+D5+D6 superset line).
- Measured tree: `codex/factor-retest` tip carrying the D6 erratum #4 fix
  cherry-pick (`c8d0043`, from 4b065c0) and the t58-reviewed test_runspec
  pin carry (`97d6b1d`, hunk from aefb71d) — the exact tip is recorded in
  `docs/t18_factor_retest_measurement_log.md` and in `metrics.json`
  provenance.
- Dataset: `D:\minequant\AlphaGPT-d2\data\ashare_v2.duckdb`; dataset_id
  recomputed read-only at run time via
  `ashare_data.manifest.build_v2_dataset_manifest` (p17 §4.4) and recorded
  in `metrics.json` → `provenance.dataset_id_computed_unpersisted`.
- Legacy database (universe/calendar/stocks/legacy fundamentals):
  `D:\minequant\AlphaGPT\data\ashare.duckdb` via `ASHARE_DUCKDB_PATH`
  (read-only; zero writes, AGENTS §5.6).

## Reproduction

    $env:ASHARE_DUCKDB_PATH = "D:\minequant\AlphaGPT\data\ashare.duckdb"
    python scripts/factor_inventory_audit.py --generation v2 `
        --v2-duckdb-path "D:\minequant\AlphaGPT-d2\data\ashare_v2.duckdb"

Append-only notice: `metrics.json` and the measurement log are evidence —
do not edit or regenerate them; a re-run writes to a new directory or is
recorded as a separate run in the measurement log.
