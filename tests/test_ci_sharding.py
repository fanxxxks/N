"""CI sharding + warning-merge contracts (PR4/D1+D2, 2026-08-31 plan).

Two single implementations under test:

* ``scripts/check_test_shards.py`` — the only authority for how the CI
  test job splits ``tests/test_*.py`` across matrix legs.  The
  "shard union == full set" property is fail-closed: any new test file
  that is not assigned to exactly one shard breaks ``--check`` (and
  therefore CI) instead of silently never running.
* ``scripts/ci_warning_merge.py`` — machine-enforces the warning
  accounting criterion fixed in PR3: the merge job fails on any
  *net-new* line in a shard's pytest warnings summary versus the
  committed baseline (``docs/ci_warning_baseline.json``), so new
  warning categories cannot slip into CI unreviewed (AGENTS §10.1).

The CI workflow itself is contract-checked here too: the matrix shard
names and the guard/emit invocations must stay in sync with the script
(AGENTS §10.2 forbids drifting gate command lists).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARD_SCRIPT = ROOT / "scripts" / "check_test_shards.py"
MERGE_SCRIPT = ROOT / "scripts" / "ci_warning_merge.py"
BASELINE = ROOT / "docs" / "ci_warning_baseline.json"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses needs the module registered
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- D1 shards


def test_shard_script_exists() -> None:
    assert SHARD_SCRIPT.exists(), "CI shard authority script is missing"


def test_shard_union_covers_all_test_files_exactly() -> None:
    module = _load_script(SHARD_SCRIPT, "check_test_shards")
    files = set(module.discover_test_files())
    assert files, "no test files discovered"
    assigned = [f for shard in module.SHARDS.values() for f in shard]
    assert len(assigned) == len(set(assigned)), "a test file is in two shards"
    assert set(assigned) == files, "shard union != tests/test_*.py"


def test_shard_validate_reports_omission_overlap_and_unknown() -> None:
    module = _load_script(SHARD_SCRIPT, "check_test_shards")
    # omission
    errors = module.validate({"a": ("x.py",)}, files={"x.py", "y.py"})
    assert any("y.py" in e for e in errors)
    # overlap
    errors = module.validate({"a": ("x.py",), "b": ("x.py",)}, files={"x.py"})
    assert any("x.py" in e for e in errors)
    # unknown file inside a shard (renamed/deleted file left behind)
    errors = module.validate({"a": ("x.py", "ghost.py")}, files={"x.py"})
    assert any("ghost.py" in e for e in errors)


def test_emit_prints_existing_relative_test_paths() -> None:
    module = _load_script(SHARD_SCRIPT, "check_test_shards")
    first_shard = next(iter(module.SHARDS))
    result = subprocess.run(
        [sys.executable, str(SHARD_SCRIPT), "--emit", first_shard],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    emitted = result.stdout.split()
    assert emitted, "--emit printed nothing"
    for rel in emitted:
        assert rel.startswith("tests/test_"), rel
        assert (ROOT / rel).exists(), rel


def test_emit_rejects_unknown_shard() -> None:
    result = subprocess.run(
        [sys.executable, str(SHARD_SCRIPT), "--emit", "no-such-shard"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0


def test_ci_matrix_and_guard_stay_in_sync_with_script() -> None:
    module = _load_script(SHARD_SCRIPT, "check_test_shards")
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    matrix_line = next(
        (line for line in text.splitlines() if "shard: [" in line),
        "",
    )
    assert matrix_line, "CI matrix shard list is missing"
    for name in module.SHARDS:
        assert name in matrix_line, f"CI matrix is missing shard {name}"
    assert "check_test_shards.py --check" in text, "union guard step missing"
    assert "check_test_shards.py --emit" in text, "shard emit invocation missing"


# -------------------------------------------------------------- D2 warnings


def test_warning_merge_script_exists() -> None:
    assert MERGE_SCRIPT.exists(), "CI warning merge script is missing"


def test_warning_merge_detects_net_new_warning_lines(tmp_path: Path) -> None:
    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "tests/test_a.py: 2 warnings",
            "  C:/env/osqp/interface.py:405: PendingDeprecationWarning: boom",
            "    warnings.warn(",
        ]
    )
    log = tmp_path / "pytest-shard-a.log"
    log.write_text(text + "\n", encoding="utf-8")
    baseline = {
        "total_warnings": 2,
        "section_lines": sorted(module.parse_warnings_section(text)),
    }
    run = module.compare(
        {log.name: module.parse_warnings_section(text)}, baseline
    )
    assert run.net_new == [], "identical section must not be net-new"

    drifted = text + "  C:/env/newpkg/mod.py:9: SomeNewWarning: fresh\n"
    log.write_text(drifted, encoding="utf-8")
    run2 = module.compare(
        {log.name: module.parse_warnings_section(drifted)}, baseline
    )
    assert any("SomeNewWarning" in line for line in run2.net_new), (
        "net-new warning lines must fail the merge job"
    )


def test_warning_merge_requires_every_shard_log(tmp_path: Path) -> None:
    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    baseline = {"total_warnings": 0, "section_lines": []}
    with pytest.raises(SystemExit):
        module.compare({}, baseline)


def test_committed_baseline_is_structurally_valid() -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["total_warnings"] > 0
    assert payload["section_lines"], "baseline warnings section is empty"
    assert "provenance" in payload and "4156de4" in payload["provenance"]
