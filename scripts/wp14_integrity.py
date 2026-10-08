"""CLI for WP14 (p50) experiment integrity: plan materialization and the
later precheck / reconcile / status / errata stages.

Subcommand + exit-code contract (C11 style; the subcommand shape is a
WP14 task requirement — the repo's single-command convention is
deliberately deviated from here and this is disclosed in the delivery
notes):

    wp14_integrity.py materialize --experiment <id> [--config PATH]
                                  [--repo-root PATH] [--out PATH]
    wp14_integrity.py precheck    --plan PATH
    wp14_integrity.py reconcile   --plan PATH --report PATH
                                  [--run-meta PATH] [--out PATH]
    wp14_integrity.py status      --plan PATH --report PATH
                                  [--run-meta PATH] [--out PATH]
    wp14_integrity.py errata      --plan PATH --report PATH
                                  --out-dir data/r1/r01_slow_baseline

Exit codes: 0 = output produced (for reconcile/status: classification
COMPLETED is 0 — findings are legitimate output, not failures); 1 = a
PlanValidationError / IntegrityRefusal, precheck issues, or a stubbed
subcommand not yet landed in WP14; 2 = usage error.

Stage status: ``materialize`` and ``precheck`` are live (precheck wired
to the R2 module by p56/WP18 — it was the last stubbed writer);
``reconcile``, ``status`` and ``errata`` are registered stubs that land
with DESIGN §3.3-§3.5/§7.  The write-surface guard: this CLI never
writes under ``AlphaGPT/data/`` — only the future ``errata``
subcommand may, and only NEW files inside ``data/r1/r01_slow_baseline``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_model.experiment_integrity.plan import (  # noqa: E402
    materialize_plan,
    plan_to_json,
)
from ashare_model.experiment_integrity.types import (  # noqa: E402
    EXPERIMENT_IDS,
    PlanValidationError,
)

#: The stub note shared by every subcommand that has not landed yet.
_STUB_TEMPLATE = (
    "wp14 subcommand '{name}' is not implemented yet: it lands with "
    "WP14 DESIGN {section}. This stub refuses (exit 1) rather than "
    "silently doing nothing."
)


def _refuse_output_under_data(out_path: Path) -> None:
    """The write-surface guard for every non-errata output: nothing the
    CLI writes may land under ``AlphaGPT/data/`` (red line: the only
    permitted data/ writes are NEW errata files inside
    ``data/r1/r01_slow_baseline``, and only via the errata stage)."""

    resolved = out_path.resolve()
    data_root = _ROOT / "data"
    try:
        resolved.relative_to(data_root)
    except ValueError:
        return
    raise PlanValidationError(
        "output path is inside AlphaGPT/data/ — the integrity CLI never "
        "writes there (only the errata stage may add NEW files inside "
        f"data/r1/r01_slow_baseline)\n(locator=--out:{out_path})"
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _cmd_materialize(args: argparse.Namespace) -> int:
    plan = materialize_plan(
        args.experiment,
        args.config,
        repo_root=args.repo_root,
    )
    default_out = (
        _ROOT.parent / "reports" / "wp14" / "plans" / f"{args.experiment}_plan.json"
    )
    out_path = Path(args.out) if args.out else default_out
    _refuse_output_under_data(out_path)
    written = _write_json(out_path, plan_to_json(plan))
    print(
        json.dumps(
            {
                "experiment_id": plan.experiment_id,
                "arms": len(plan.arms),
                "folds": len(plan.folds),
                "expected_row_count": plan.expected_row_count,
                "output": str(written),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_precheck(args: argparse.Namespace) -> int:
    """p56/WP18: wire the precheck subcommand to the landed R2 module
    (``run_precheck`` — live since WP14 with its own test family; the
    CLI stub was the only missing piece).  Exit 0 iff the plan has zero
    issues; the issue list is the report, written to --out when given."""

    from ashare_model.experiment_integrity.plan import plan_from_json
    from ashare_model.experiment_integrity.precheck import run_precheck

    plan_path = Path(args.plan)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"wp14_integrity: cannot read plan: {exc}", file=sys.stderr)
        return 1
    try:
        plan = plan_from_json(payload)
    except PlanValidationError as exc:
        print(f"wp14_integrity: refused: {exc}", file=sys.stderr)
        return 1
    report = run_precheck(plan)
    report_payload = {
        "precheck_schema_version": 1,
        "experiment_id": plan.experiment_id,
        "plan_path": str(plan_path),
        "ok": report.ok,
        "checks_run": list(report.checks_run),
        "expected_row_count": plan.expected_row_count,
        "issues": [
            {
                "code": issue.code,
                "experiment_id": issue.experiment_id,
                "arm": issue.arm,
                "member": issue.member,
                "locator": issue.locator,
                "detail": issue.detail,
            }
            for issue in report.issues
        ],
    }
    if args.out:
        out_path = Path(args.out)
        _refuse_output_under_data(out_path)
        _write_json(out_path, report_payload)
    print(
        json.dumps(
            {
                "experiment_id": plan.experiment_id,
                "ok": report.ok,
                "issues": len(report.issues),
                **({"output": str(args.out)} if args.out else {}),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.ok else 1


def _stub(name: str, section: str):
    def _run(args: argparse.Namespace) -> int:
        print(_STUB_TEMPLATE.format(name=name, section=section), file=sys.stderr)
        return 1

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wp14_integrity.py",
        description=(
            "WP14 (p50) experiment integrity: materialize run plans and "
            "(in later stages) precheck, reconcile, classify and write "
            "errata for the frozen r1 artifacts — read-only on data/**"
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="materialize the R1 run plan of one experiment to JSON",
    )
    materialize.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENT_IDS,
    )
    materialize.add_argument(
        "--config",
        default=None,
        help="override the experiment's frozen config path",
    )
    materialize.add_argument(
        "--repo-root",
        default=None,
        help="repository root (default: inferred from this file)",
    )
    materialize.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: reports/wp14/plans/<id>_plan.json)",
    )
    materialize.set_defaults(func=_cmd_materialize)

    precheck = subparsers.add_parser(
        "precheck", help="run the R2 load-time checks on a materialized plan"
    )
    precheck.add_argument("--plan", required=True)
    precheck.add_argument(
        "--out",
        default=None,
        help="write the precheck report JSON here (default: stdout summary only)",
    )
    precheck.set_defaults(func=_cmd_precheck)

    reconcile = subparsers.add_parser(
        "reconcile", help="reconcile a report against its plan (R3)"
    )
    reconcile.add_argument("--plan", required=True)
    reconcile.add_argument("--report", required=True)
    reconcile.add_argument("--run-meta", default=None)
    reconcile.add_argument("--out", default=None)
    reconcile.set_defaults(func=_stub("reconcile", "§3.3"))

    status = subparsers.add_parser(
        "status", help="three-layer status classification of a report (R4/R5)"
    )
    status.add_argument("--plan", required=True)
    status.add_argument("--report", required=True)
    status.add_argument("--run-meta", default=None)
    status.add_argument("--out", default=None)
    status.set_defaults(func=_stub("status", "§3.4/§5"))

    errata = subparsers.add_parser(
        "errata",
        help="write ERRATUM.md + errata.json into data/r1/r01_slow_baseline (R6)",
    )
    errata.add_argument("--plan", required=True)
    errata.add_argument("--report", required=True)
    errata.add_argument(
        "--out-dir",
        default=str(_ROOT / "data" / "r1" / "r01_slow_baseline"),
    )
    errata.set_defaults(func=_stub("errata", "§7"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PlanValidationError as exc:
        print(f"wp14_integrity: refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
