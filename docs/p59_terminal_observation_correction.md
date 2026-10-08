# P59 terminal observation correction before postprocessing

Final output review found summary v1's `actual_exposure_final` took the last
entry-open exposure (2025-12-30), while terminal assets are the last exit-open
mark (2025-12-31). Money, returns, drawdowns, costs, fills and average entry
exposures are unaffected. The snapshot/daily exposure observations correctly
refer to their existing entry_date; their meaning must not change.

Fix only terminal summary exposure to `(terminal assets - final cash) /
terminal assets`, retaining the prior value explicitly as
`actual_exposure_last_entry`. Use the single local summary version owner and
bump 1 to 2 for this reporting semantic correction; no strategy/execution/
constructor/protocol/reward/artifact-boundary versions change. Version 1 remains
preserved local evidence, with the terminal-field correction clearly disclosed.

RED: 100k account, 99k filled notional, 150.49 costs, 849.51 cash; next-open
price doubles. Terminal NAV=198849.51, exposure=198000/198849.51, differing
from entry exposure=99000/99849.51. This independent oracle must fail on v1.

The six original engine runs remain the only account runs. Correct the six
already-recorded summaries using their audited terminal assets and cash,
write new `results_corrected.json` and `summary_corrected.csv` without
overwriting v1 results or raw CSVs, and record the original engine SHA and
correction SHA distinctly. Existing result tables do not display the affected
field; table amounts and conclusions remain unchanged. No research rerun,
signal/config/parameter/input changes, new budget or dependency is permitted.
