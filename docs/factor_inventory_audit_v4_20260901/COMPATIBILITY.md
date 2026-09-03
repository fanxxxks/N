# Compatibility note — audit harness relocated (IP-14, 2026-09-03)

The runnable audit code that used to live in this directory has been
consolidated into a single parameterized tool:

    scripts/factor_inventory_audit.py

* This directory's historical script `audit_run_v4.py` (t5, 2026-09-01,
  73-feature post-P9 v4 audit) is retired; its exact producing copy is
  frozen in git history at commit `d6a034d` and remains retrievable at
  this path in any commit up to and including `8641e1f`.
* To reproduce this directory's `metrics.json` schema, run:
  `python scripts/factor_inventory_audit.py --generation v4`
  (output defaults to this directory; override with `--out-dir`).
  The v4 profile asserts the current 73-feature vocabulary
  (`FEATURE_NAMES` tail == `_FEATURE_NAMES_V4`,
  `DEPRECATED_FEATURE_NAMES ⊆ FEATURE_NAMES`).
* `metrics.json` and `audit_report_v4.md` in this directory are untouched
  historical measurement evidence (append-only; never regenerate into
  this directory without an explicit evidence task).
* Per `docs/p13_fundamental_fields_contract.md` §3.3 the historical audit
  scripts are never edited; adjudication re-tests run the new tool (a new
  script), never the frozen evidence.
