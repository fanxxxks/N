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


def _ci_web_job_steps() -> str:
    """The ``web`` job block of ci.yml (from its header to the next
    2-space-indented job key)."""

    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "  web:")
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1:], start + 1)
            if line.startswith("  ") and not line.startswith("   ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


WEB_GATE_COMMANDS = (
    "npm ci",
    "npm ls --depth=0",
    "npm test",
    "npm run lint",
    "npm run build",
)


def test_onboarding_web_gates_match_ci() -> None:
    """§9.2 must mirror the CI web job: clean install, tree check, unit
    tests, type-check lint, build (p31 B02 promoted ``npm test`` /
    ``npm run lint`` from local-only conveniences to CI gates; both the
    workflow and the onboarding mirror must carry them)."""

    section = _onboarding_recommended_commands()
    web_job = _ci_web_job_steps()
    for fragment in WEB_GATE_COMMANDS:
        assert fragment in section, f"missing from onboarding §9.2: {fragment}"
        assert fragment in web_job, f"missing from ci.yml web job: {fragment}"


def test_onboarding_has_no_standalone_tsc_step() -> None:
    """No §9.2 line may run a bare ``tsc`` directly: the lint gate is the
    ``npm run lint`` script (p31 B02 moved it into the CI web job, so the
    script — not a hand-typed compiler invocation — is the single command
    both locals and CI execute).  Mentioning the equivalence in a comment
    is fine; a standalone ``tsc`` command line is the 2026-08-31 drift."""

    offenders = [
        line.strip()
        for line in _onboarding_recommended_commands().splitlines()
        if line.strip().startswith("tsc")
    ]
    assert not offenders, f"standalone tsc command in onboarding §9.2: {offenders}"


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
