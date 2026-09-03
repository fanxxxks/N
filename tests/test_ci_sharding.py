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


def test_parse_excludes_singular_group_count_header() -> None:
    """IP-09 (04-TC-05): singular group-count headers must be excluded too.

    Runtime evidence (t1 serial gate @ 28bfefb, logs/gate_28bfefb.txt):
    the train All-NaN block dropped from two instances to one, so pytest
    emitted ``tests/test_train.py: 1 warning`` — the plural-only
    COUNT_HEADER_RE let that header escape into the comparison as a
    net-new line (pure instance multiplicity, non-comparable per the
    PR3 criterion).  Flip coverage: 0 (no header), 1 (singular), 2
    (plural) must all leave only the location/message lines.
    """

    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    singular_text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "tests/test_train.py: 1 warning",
            "  D:\\minequant\\AlphaGPT\\ashare_model\\reward.py:629: "
            "RuntimeWarning: All-NaN slice encountered",
            "    capacity_full[:, output_col] = np.nanmax(util, axis=1)",
            "1475 passed, 5 skipped, 632 warnings in 1521.33s (0:25:21)",
        ]
    )
    section = module.parse_warnings_section(singular_text)
    assert not any("test_train.py: 1 warning" in line for line in section), (
        "singular count header carries instance multiplicity and must be excluded"
    )
    assert any("reward.py:629" in line for line in section), (
        "the location line itself must stay in the section"
    )
    assert not any("passed," in line for line in section), (
        "the totals line must stay excluded"
    )

    # 0-case: a block with no count header at all keeps its location lines.
    headerless_text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "  site-packages/osqp/interface.py:405: PendingDeprecationWarning: boom",
            "    warnings.warn(",
            "1475 passed, 5 skipped, 632 warnings in 1521.33s (0:25:21)",
        ]
    )
    section_headerless = module.parse_warnings_section(headerless_text)
    assert any("interface.py:405" in line for line in section_headerless)

    # 2->1 flip: the plural spelling keeps behaving as before.
    plural_text = singular_text.replace(
        "tests/test_train.py: 1 warning", "tests/test_train.py: 2 warnings"
    )
    section_plural = module.parse_warnings_section(plural_text)
    assert not any("test_train.py: 2 warnings" in line for line in section_plural)
    assert any("reward.py:629" in line for line in section_plural)


def test_parse_summary_totals_accepts_singular_warning() -> None:
    """IP-09 (04-TC-05): pytest writes ``1 warning`` (singular) when a run
    produced exactly one warning; the plural-only SUMMARY_RE failed to
    match such a totals line at all — parse_summary_totals raised
    ValueError and the totals line leaked into the section (it never
    terminated it).  Flip coverage: 0/1/2 warnings and skipped variants.
    """

    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    assert module.parse_summary_totals("3 passed in 0.50s") == (3, 0, 0)
    assert module.parse_summary_totals("5 passed, 2 skipped in 0.50s") == (5, 2, 0)
    assert module.parse_summary_totals("1 passed, 1 warning in 0.50s") == (1, 0, 1)
    assert module.parse_summary_totals(
        "1 passed, 1 skipped, 1 warning in 0.50s"
    ) == (1, 1, 1)
    assert module.parse_summary_totals("2 passed, 2 warnings in 0.50s") == (2, 0, 2)
    assert module.parse_summary_totals(
        "1475 passed, 5 skipped, 632 warnings in 1521.33s (0:25:21)"
    ) == (1475, 5, 632)


def test_section_end_terminates_on_singular_totals_line() -> None:
    """IP-09: the run-varying totals line must terminate the warnings
    section regardless of its warning-count spelling; with the plural-only
    SUMMARY_RE a ``1 warning`` totals line leaked into the section and
    every such comparison went net-new by construction."""

    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "  site-packages/osqp/interface.py:405: PendingDeprecationWarning: boom",
            "    warnings.warn(",
            "1 passed, 1 warning in 0.50s",
        ]
    )
    section = module.parse_warnings_section(text)
    assert not any("passed," in line for line in section), (
        "a singular-spelled totals line must still terminate the section"
    )
    assert any("interface.py:405" in line for line in section)


def test_write_baseline_dedupes_section_lines(tmp_path: Path) -> None:
    """IP-09: the committed baseline is a reviewable artifact; the t1
    serial gate produced repeated normalized lines (the same node header
    once per warning block — 7 headers x 2 blocks — and
    ``tests/test_core.py: 1 warning`` twice in the leaked predecessor).
    compare() is set-based so duplicates never changed enforcement, but
    write_baseline must emit the deduplicated set so the artifact itself
    carries no multiplicity."""

    module = _load_script(MERGE_SCRIPT, "ci_warning_merge")
    text = "\n".join(
        [
            "============================== warnings summary ===============================",
            "tests/test_diagnostics.py::test_factor_report_smoke",
            "  site-packages/numpy/lib/_function_base_impl.py:3036: RuntimeWarning: x",
            "tests/test_diagnostics.py::test_factor_report_smoke",
            "  site-packages/numpy/lib/_function_base_impl.py:3037: RuntimeWarning: y",
            "1 passed, 1 warning in 0.50s",
        ]
    )
    log = tmp_path / "pytest_serial.log"
    log.write_text(text + "\n", encoding="utf-8")
    out = tmp_path / "baseline.json"
    assert module.write_baseline(out, log, "test provenance") == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    lines = payload["section_lines"]
    assert len(lines) == len(set(lines)), "section_lines must be deduplicated"
    assert lines == sorted(set(lines)), "section_lines must stay sorted"
    assert payload["total_warnings"] == 1, (
        "the singular-spelled totals must count as one warning"
    )


def test_ci_warning_merge_step_pins_matrix_leg_count() -> None:
    """IP-09 (04-TC-06): the merge job must pass --expect with the matrix
    leg count; without it a missing shard log silently weakens the union
    (the R3-F3b fail-closed fix existed in the script but was never wired
    into CI)."""

    text = CI_WORKFLOW.read_text(encoding="utf-8")
    merge_line = next(
        (
            line
            for line in text.splitlines()
            if "ci_warning_merge.py --check" in line
        ),
        "",
    )
    assert merge_line, "warning merge step is missing from ci.yml"
    assert "--expect 4" in merge_line, (
        "the merge step must fail closed on a missing matrix leg (--expect 4)"
    )


def test_committed_baseline_is_structurally_valid() -> None:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["total_warnings"] > 0
    assert payload["section_lines"], "baseline warnings section is empty"
    # IP-09: the baseline source moved from the 4156de4 PR3 gate to the
    # 28bfefb t1 serial gate (regeneration required by the comparator's
    # singular-header fix); the assertion keeps pinning the exact source
    # sha of the current baseline (same strength, new requirement).
    assert "provenance" in payload and "28bfefb" in payload["provenance"]


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
