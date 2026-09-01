"""Drift guard: AGENTS.md stays under a growth budget.

The root AGENTS.md is injected into every agent session, so unbounded
growth dilutes rule density and attention.  This guard pins a 36 KiB
working cap: new rules should pay for themselves (deduplicate or move
detail behind pointers) rather than accrete.

The cap is a growth bound, not a compression mandate (user decision,
2026-09-01: content completeness outweighs size).  Note the honest
trade-off it records: the file exceeds the Codex default
``project_doc_max_bytes`` = 32 KiB, so default Codex CLI configurations
will truncate it unless the budget is raised; AGENTS.md carries the
same warning for readers.

Size is measured on the raw UTF-8 bytes (``Path.stat().st_size``) —
the encoding stored on disk and consumed by the tools.  Counting
UTF-16 code units or characters instead would misreport CJK-heavy
text by up to 3x.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = ROOT / "AGENTS.md"

# 36 KiB growth cap (user-approved 2026-09-01).  Exceeding 32 KiB is
# accepted with an in-file Codex warning; 36 KiB is the hard ceiling.
MAX_BYTES = 36 * 1024


def test_agents_md_stays_under_growth_budget() -> None:
    size = AGENTS_PATH.stat().st_size
    assert size <= MAX_BYTES, (
        f"AGENTS.md is {size} bytes (> {MAX_BYTES}): the root instruction "
        "file is injected into every session and unbounded growth dilutes "
        "rule density. Deduplicate or move detail behind pointers instead "
        "of growing the file."
    )
