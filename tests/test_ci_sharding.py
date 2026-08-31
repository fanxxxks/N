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
import re
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


def test_parse_excludes_final_summary_line() -> None:
    """Regression (PR4 final-candidate verification): the summary line
    (\"N passed, ... in T\") is run-varying and must never enter the
    warnings section, or every comparison would be net-new by
    construction."""

    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "tests/test_a.py: 1 warning",
            "  C:/env/osqp/interface.py:405: PendingDeprecationWarning: boom",
            "    warnings.warn(",
            "1342 passed, 5 skipped, 615 warnings in 272.18s (0:04:32)",
        ]
    )
    section = module.parse_warnings_section(text)
    assert section, "warnings summary must be parsed"
    assert not any("passed," in line for line in section), (
        "the run-varying totals line must be excluded from the section"
    )


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


def test_warning_merge_ignores_group_count_headers(tmp_path: Path) -> None:
    """Regression (2026-08-31 principle-7 gate): instance-count headers are
    not comparable across xdist and serial shapes.

    The once-per-process warning duplication documented in PR3 manifests
    ONLY as a group-count header delta (serial '7 warnings' vs xdist
    '8 warnings') while every location/message line - the actual warning
    kinds - stays identical.  The comparator implements the registered
    criterion (deduplicated category/message/location set), so count
    headers must be excluded from both sides; otherwise every local
    `-n auto` gate fails on the documented delta.
    """

    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    location_lines = [
        "  C:/env/osqp/interface.py:405: PendingDeprecationWarning: boom",
        "    warnings.warn(",
    ]
    serial_text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "tests/test_evaluation.py: 7 warnings",
            *location_lines,
            "1347 passed, 5 skipped, 614 warnings in 730.02s (0:12:10)",
        ]
    )
    xdist_text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "tests/test_evaluation.py: 8 warnings",
            *location_lines,
            "1347 passed, 5 skipped, 615 warnings in 386.87s (0:06:26)",
        ]
    )
    baseline = {
        "total_warnings": 614,
        "section_lines": sorted(module.parse_warnings_section(serial_text)),
    }
    run = module.compare(
        {"pytest-xdist.log": module.parse_warnings_section(xdist_text)}, baseline
    )
    assert run.net_new == [], (
        "count-header-only deltas must not be net-new (instance multiplicity "
        "is documented as non-comparable)"
    )
    assert run.disappeared == []


def test_committed_baseline_is_structurally_valid() -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["total_warnings"] > 0
    assert payload["section_lines"], "baseline warnings section is empty"
    assert "provenance" in payload and "4156de4" in payload["provenance"]


def test_baseline_lines_are_environment_independent() -> None:
    """R3-F1: baseline lines carry no machine-specific path prefixes."""

    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    for line in payload["section_lines"]:
        assert not re.match(r"^[A-Za-z]:", line), f"drive-letter path leaked: {line}"
        assert "miniconda" not in line, f"interpreter path leaked: {line}"
        assert not line.startswith("D:/") and not line.startswith("/home/"), line


def test_parse_normalizes_location_prefixes() -> None:
    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    windows_local = (
        "  D:\\minequant\\AlphaGPT\\ashare_model\\reward.py:615: "
        "RuntimeWarning: All-NaN slice encountered"
    )
    third_party = (
        "  C:\\ProgramData\\miniconda3\\Lib\\site-packages\\osqp\\interface.py:405: "
        "PendingDeprecationWarning: boom"
    )
    linux_ci = (
        "  /home/runner/work/N/N/ashare_model/data_loader.py:86: "
        "UniverseDevelopmentFallbackWarning: dev fallback"
    )
    assert module._normalize_line(windows_local) == (
        "ashare_model/reward.py:615: RuntimeWarning: All-NaN slice encountered"
    )
    assert module._normalize_line(third_party) == (
        "site-packages/osqp/interface.py:405: PendingDeprecationWarning: boom"
    )
    assert module._normalize_line(linux_ci) == (
        "ashare_model/data_loader.py:86: UniverseDevelopmentFallbackWarning: dev fallback"
    )
    assert module._normalize_line("tests/test_a.py: 2 warnings") == "tests/test_a.py: 2 warnings"


def test_discover_is_recursive(tmp_path: Path) -> None:
    module = _load_script(SHARD_SCRIPT, "check_test_shards")
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_top.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "sub").mkdir()
    (tmp_path / "tests" / "sub" / "test_nested.py").write_text("", encoding="utf-8")
    files = module.discover_test_files(root=tmp_path)
    assert files == {"test_top.py", "sub/test_nested.py"}


def test_check_expect_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"total_warnings": 0, "section_lines": []}), encoding="utf-8")
    log = tmp_path / "pytest-shard-a.log"
    log.write_text("nothing", encoding="utf-8")
    with pytest.raises(SystemExit):
        module.check(baseline, [log], expect=4)
