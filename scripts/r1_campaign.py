"""B03 (p32) R1 campaign CLI: the resumable, identity-bound, budgeted
driver that replaces the ``.done``-marker skip logic of the legacy
``scripts/r1_campaign.ps1`` (audit F09).

The R1 manifest (p21 §9 runbook semantics, preserved) is: one
engineering calibration leg (step 0, the preregistered p21 §5
S-fallback measurement) followed by the five runnable config legs.  The
default phase stays calibration-only; ``--run-configs`` proceeds to the
config legs.  The S=512-vs-256 decision remains a RESEARCH decision
taken from the calibration evidence before any formal config run — this
driver performs no adjudication and mints no formal research identity
(every leg runs as an explicit engineering probe, p27 I2).

Exit codes (p32 §3.4): 0 completed, 1 any leg failed / budget stop,
130 user abort, 2 operational refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Importable regardless of the caller's sys.path (repo script convention).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_model.campaign_executor import (
    EXIT_ABORTED,
    EXIT_REFUSED,
    CampaignExecutorError,
    CampaignLeg,
    CampaignLockedError,
    CampaignManifest,
    CampaignRunner,
)

CALIBRATION_LEG = "calibration"
RUN_CONFIG_LEGS = (
    "r1_medium_a",
    "r1_medium_ab",
    "r1_slow_ab",
    "r1_short_a",
    "r1_unified_base",
)


def build_r1_manifest(repo_root: Path) -> CampaignManifest:
    """The R1 campaign manifest: the calibration leg plus the five
    runnable config legs (p21 §4/§9).  Budgets are manifest-frozen
    (p32 §3.3); ``None`` = explicitly uncapped until a campaign freezes
    real values."""

    def _leg(leg_id: str, config_name: str | None = None, extra_args: tuple[str, ...] = ()) -> CampaignLeg:
        return CampaignLeg(
            leg_id=leg_id,
            config=repo_root / "configs" / f"{config_name or leg_id}.yaml",
            output=repo_root / "data" / "r1" / leg_id / "protocol_result.json",
            ledger=repo_root / "data" / "r1" / "ledger.jsonl",
            extra_args=extra_args,
        )

    return CampaignManifest(
        name="r1",
        legs=(
            _leg(CALIBRATION_LEG, config_name="r1_calibration", extra_args=("--folds", "1", "--seeds", "42")),
            *[_leg(leg_id) for leg_id in RUN_CONFIG_LEGS],
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="B03 R1 campaign driver (resumable, identity-bound, "
        "budgeted; docs/p32_b03_resumable_execution_budget_draft.md)"
    )
    parser.add_argument(
        "--run-configs",
        action="store_true",
        help="run the five config legs after calibration (p21 phase 2)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run terminal-failed legs in NEW runs (old evidence is "
        "preserved); without this flag failed legs are skipped (p32 §3.5)",
    )
    parser.add_argument(
        "--clear-stale-lock",
        action="store_true",
        help="explicitly clear a stale campaign lock after verifying the "
        "holder pid is dead (p32 §3.6)",
    )
    parser.add_argument(
        "--state-dir",
        default="data/r1/campaign",
        help="campaign state directory (default: data/r1/campaign)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    manifest = build_r1_manifest(repo_root)
    leg_ids = (
        None if args.run_configs else [CALIBRATION_LEG]
    )
    runner = CampaignRunner(
        manifest,
        repo_root / args.state_dir,
        repo_root=repo_root,
        clear_stale_lock=args.clear_stale_lock,
        retry_failed=args.retry_failed,
    )
    try:
        summary = runner.run(leg_ids=leg_ids)
    except CampaignLockedError as exc:
        print(f"r1_campaign: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except CampaignExecutorError as exc:
        print(f"r1_campaign: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        print("r1_campaign: aborted by user", file=sys.stderr)
        return EXIT_ABORTED
    print(
        f"r1_campaign: status={summary['campaign_status']} "
        f"exit={summary['exit_code']} counts={summary['counts']}"
    )
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
