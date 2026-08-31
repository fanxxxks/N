"""CI warning-merge authority (PR4/D2 of the 2026-08-31 test-runtime plan).

Machine-enforces the warning accounting criterion fixed in PR3: the CI
merge job compares the *union of shard warnings-summary sections* against
a committed baseline (``docs/ci_warning_baseline.json``) and fails on any
**net-new** line — i.e. a warning kind, message template, or call site
that did not exist in the reference serial run (AGENTS §10.1: new
warnings are a regression signal and must be explained item-by-item
before the baseline is regenerated).

Line-level granularity (exact text of the summary lines) is deliberate:
it is the same comparison that proved decisive during PR3's parity
trials, it needs no third-party plugin, and the committed baseline is a
reviewable diffable artifact with its provenance recorded.

Usage:
    python scripts/ci_warning_merge.py --check \\
        --baseline docs/ci_warning_baseline.json --logs pytest-*.log
    python scripts/ci_warning_merge.py --write-baseline OUT.json \\
        --from pytest_serial_gate.log --provenance "serial gate @ <sha> ..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

START_MARK = "warnings summary"
END_MARKS = ("slowest", "durations:", "short test summary")
SUMMARY_RE = re.compile(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) warnings)? in ")

# In-repo package/tool anchors: a warnings-summary location is normalized
# to start at the last of these, so a Windows dev tree (D:\...\),
# a Linux CI checkout (/home/runner/work/...) and any other machine
# produce identical lines (R3-F1).
_REPO_ANCHORS = (
    "ashare_data",
    "ashare_model",
    "ashare_portfolio",
    "ashare_trading",
    "tests",
    "scripts",
    "webapi",
    "webui",
)


def _normalize_line(line: str) -> str:
    """Make a warnings-summary line environment-independent (R3-F1).

    Separators are unified to '/', third-party locations are anchored at
    their ``site-packages`` segment, and in-repo locations are re-anchored
    at their package/tool prefix.  Both the committed baseline and every
    compared run go through this function, so the comparison is exact yet
    machine-independent.
    """

    out = line.strip().replace("\\", "/")
    idx = out.find("site-packages/")
    if idx != -1:
        return "site-packages/" + out[idx + len("site-packages/"):]
    for anchor in _REPO_ANCHORS:
        idx = out.rfind(anchor + "/")
        if idx != -1:
            return out[idx:]
    return out


def parse_warnings_section(text: str) -> list[str]:
    """Normalized lines of the pytest warnings-summary section.

    The section excludes the run-varying totals line (\"N passed, ... in
    T\") — it must never enter the comparison (PR4 final-candidate
    verification regression) — and every line is environment-normalized
    (R3-F1), so the committed baseline is machine-independent.
    """

    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if START_MARK in line), None)
    if start is None:
        return []
    end = len(lines)
    for i in range(start + 1, len(lines)):
        # Stop at the next section separator, a duration table, the short
        # summary - and in particular at the run-varying totals line.
        if any(mark in lines[i] for mark in END_MARKS) or SUMMARY_RE.search(lines[i]):
            end = i
            break
    return [_normalize_line(line) for line in lines[start:end] if line.strip()]


def parse_summary_totals(text: str) -> tuple[int, int, int]:
    """(passed, skipped, warnings) from the final pytest summary line."""

    for line in reversed(text.splitlines()):
        match = SUMMARY_RE.search(line)
        if match:
            passed = int(match.group(1))
            skipped = int(match.group(2) or 0)
            warnings = int(match.group(3) or 0)
            return passed, skipped, warnings
    raise ValueError("no pytest summary line found in log")


@dataclass
class CompareResult:
    net_new: list[str] = field(default_factory=list)
    disappeared: list[str] = field(default_factory=list)


def compare(runs: dict[str, list[str]], baseline: dict) -> CompareResult:
    """Union the per-shard sections and diff against the baseline lines."""

    if not runs:
        raise SystemExit(
            "no shard logs provided; failing closed - every matrix leg must "
            "upload its pytest log or the warning baseline cannot be enforced"
        )
    union: set[str] = set()
    for lines in runs.values():
        union.update(lines)
    baseline_lines = set(baseline.get("section_lines", []))
    return CompareResult(
        net_new=sorted(union - baseline_lines),
        disappeared=sorted(baseline_lines - union),
    )


def check(baseline_path: Path, log_paths: list[Path], expect: int | None = None) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if expect is not None and len(log_paths) != expect:
        raise SystemExit(
            f"expected {expect} shard log(s) but got {len(log_paths)} - "
            "a matrix leg is missing its artifact; fail closed"
        )
    runs = {p.name: parse_warnings_section(p.read_text(encoding="utf-8")) for p in log_paths}
    missing = [p.name for p in log_paths if not runs[p.name]]
    if missing:
        print(
            "shard log(s) without a warnings summary section: "
            + ", ".join(sorted(missing))
            + " - fail closed",
            file=sys.stderr,
        )
        return 1
    result = compare(runs, baseline)

    passed = skipped = warnings = 0
    for path in log_paths:
        p, s, w = parse_summary_totals(path.read_text(encoding="utf-8"))
        passed += p
        skipped += s
        warnings += w

    print(
        f"warning merge: {len(log_paths)} shard logs, "
        f"{passed} passed / {skipped} skipped / {warnings} warnings total; "
        f"baseline {baseline.get('total_warnings')} warnings "
        f"from {baseline.get('provenance', 'unknown provenance')}"
    )
    if result.disappeared:
        print(
            "warning lines present in the baseline but absent from this run "
            "(informational, review):",
            file=sys.stderr,
        )
        for line in result.disappeared:
            print(f"  - {line}", file=sys.stderr)
    if result.net_new:
        print("NET-NEW warning lines (fail):", file=sys.stderr)
        for line in result.net_new:
            print(f"  + {line}", file=sys.stderr)
        print(
            "new warning categories must be explained item-by-item in "
            "docs/test_runtime_measurement_log.md and the baseline "
            "regenerated with --write-baseline before merging.",
            file=sys.stderr,
        )
        return 1
    print("no net-new warning lines: baseline holds")
    return 0


def write_baseline(out_path: Path, from_log: Path, provenance: str) -> int:
    text = from_log.read_text(encoding="utf-8")
    section = parse_warnings_section(text)
    _, _, warnings = parse_summary_totals(text)
    payload = {
        "provenance": provenance,
        "total_warnings": warnings,
        "section_lines": sorted(section),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({warnings} warnings, {len(section)} section lines)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="enforce the baseline")
    parser.add_argument("--baseline", type=Path, help="committed baseline JSON (required with --check)")
    parser.add_argument("--logs", nargs="+", type=Path, help="shard pytest output logs")
    parser.add_argument("--write-baseline", type=Path, metavar="OUT")
    parser.add_argument("--from", dest="from_log", type=Path, metavar="LOG")
    parser.add_argument("--provenance", default="")
    parser.add_argument(
        "--expect",
        type=int,
        help="required number of shard logs (matrix leg count); fail closed otherwise",
    )
    args = parser.parse_args(argv)

    if args.write_baseline:
        if not args.from_log:
            parser.error("--write-baseline requires --from")
        return write_baseline(args.write_baseline, args.from_log, args.provenance)
    if args.check:
        if not args.logs or not args.baseline:
            parser.error("--check requires --baseline and --logs")
        return check(args.baseline, args.logs, expect=args.expect)
    parser.error("nothing to do: pass --check or --write-baseline")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
