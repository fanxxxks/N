# Compatibility note — audit harness relocated (IP-14, 2026-09-03)

The runnable audit code that used to live in this directory has been
consolidated into a single parameterized tool:

    scripts/factor_inventory_audit.py

* This directory's historical script `audit_run.py` (t2, 2026-08-31,
  62-feature baseline) is retired; its exact producing copy is frozen in
  git history at commit `23c82f4` and remains retrievable at this path in
  any commit up to and including `8641e1f`.
* To reproduce this directory's `metrics.json` schema, run:
  `python scripts/factor_inventory_audit.py --generation v1`
  (output defaults to this directory; override with `--out-dir`).
  The v1 profile is reproducible only against the t2-era 62-feature
  vocabulary — the current vocabulary is v4 (73 features).
* `metrics.json` and `audit_report.md` in this directory are untouched
  historical measurement evidence (append-only; never regenerate into
  this directory without an explicit evidence task).
* Per `docs/p13_fundamental_fields_contract.md` §3.3 the historical audit
  scripts are never edited; adjudication re-tests run the new tool (a new
  script), never the frozen evidence.
