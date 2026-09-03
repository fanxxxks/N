"""Drift guard: validation-command lists must stay in sync across AGENTS.md
and docs/PROJECT_ONBOARDING.md.

AGENTS §10.2 (gate-sync rule in the documentation/config-changes paragraph)
requires AGENTS, CI, the local verification entry points, and related docs
to move together and forbids drifting command lists. On 2026-08-31 an audit
found exactly such drift: the onboarding recommended commands lacked the
parallel ``compileall`` invocation, ``git diff --check``, and
``npm ls --depth=0``, while listing a standalone ``tsc --noEmit`` step that
CI does not run (tsc executes inside ``npm run build`` = ``tsc -b && vite
build``). These tests pin the shared command list so the drift cannot recur
silently.

The assertions target the recommended-commands section (§9.2) only; the
historical verification record in §9.1 is intentionally left alone.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AGENTS = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
ONBOARDING = (ROOT / "docs" / "PROJECT_ONBOARDING.md").read_text(encoding="utf-8")

# §10.2 step 4, canonical form: parallel bytecode compilation (-j 0 uses all
# cores; zero test/production semantics change).
COMPILEALL_CMD = (
    "python -m compileall -j 0 -q ashare_data ashare_model "
    "ashare_portfolio ashare_trading scripts webapi"
)

# §10.2 step 3, parallel local full-suite gate.  Legal only with per-PR
# parallel-vs-serial parity evidence (counts + warnings multiset) recorded
# in docs/test_runtime_measurement_log.md; CI job commands are unchanged.
PYTEST_GATE_CMD = "python -m pytest -q tests -n auto"


def _onboarding_recommended_commands() -> str:
    """Return the §9.2 recommended-commands section (up to the next heading)."""

    start = ONBOARDING.index("### 9.2")
    next_heading = ONBOARDING.find("\n### ", start + 1)
    end = next_heading if next_heading != -1 else len(ONBOARDING)
    return ONBOARDING[start:end]


def test_agents_compileall_uses_parallel_jobs() -> None:
    """§10.2 step 4 must compile with -j 0 (all cores)."""

    assert COMPILEALL_CMD in AGENTS


def test_onboarding_python_gates_match_agents() -> None:
    """§9.2 must list the same pre-push Python gates as AGENTS §10.2."""

    section = _onboarding_recommended_commands()
    for fragment in (
        COMPILEALL_CMD,
        "python -m pytest -q tests",
        "git diff --check",
        "python -m pip check",
        "python scripts/freeze_lock.py --check",
    ):
        assert fragment in section, f"missing from onboarding §9.2: {fragment}"


def test_onboarding_web_gates_match_ci() -> None:
    """§9.2 must mirror the CI web job: clean install, tree check, build."""

    section = _onboarding_recommended_commands()
    for fragment in ("npm ci", "npm ls --depth=0", "npm run build"):
        assert fragment in section, f"missing from onboarding §9.2: {fragment}"


def test_onboarding_has_no_standalone_tsc_step() -> None:
    """tsc runs inside `npm run build`; a standalone step drifts from CI."""

    assert "tsc --noEmit" not in _onboarding_recommended_commands()


def test_local_full_suite_gate_runs_parallel() -> None:
    """§10.2 step 3 and onboarding §9.2 must use the parallel gate form."""

    assert PYTEST_GATE_CMD in AGENTS
    assert PYTEST_GATE_CMD in _onboarding_recommended_commands()


def test_docs_contains_no_executable_python() -> None:
    """IP-14 (docs hygiene): docs/ carries measurement evidence, narrative
    reports and contracts only -- runnable code lives in scripts/ (or
    proper packages).  The factor-inventory audit harnesses were
    consolidated into scripts/factor_inventory_audit.py with per-directory
    COMPATIBILITY.md pointers; this drift guard keeps docs/ .py-free."""

    py_files = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "docs").rglob("*.py")
    )
    assert not py_files, f"executable .py files under docs/: {py_files}"


def test_audit_harness_consolidation_pointers_exist() -> None:
    """IP-14: both audit evidence directories keep their COMPATIBILITY.md
    pointer to the consolidated tool, and the consolidated tool exists."""

    for marker in (
        "docs/factor_inventory_audit_20260831/COMPATIBILITY.md",
        "docs/factor_inventory_audit_v4_20260901/COMPATIBILITY.md",
    ):
        assert (ROOT / marker).is_file(), f"missing pointer: {marker}"
    assert (ROOT / "scripts" / "factor_inventory_audit.py").is_file()
    assert (ROOT / "scripts" / "limit_incidence_probe.py").is_file()
