"""P50 WP14 reconcile-stage RED/GREEN tests: expected-vs-actual set
reconciliation by unique key (R3).

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.3 (the WP14
reconcile stage; the contract file lands with the WP14
contract-writing stage per DESIGN §1 — every clause asserted here is
pinned to a DESIGN §3.3 rule and a measured fact F3/F4/F8/F9/F10/F12/
F19/F20 from reports/wp14/_recon/report_schemas.md).

Invariants (reconcile stage — the DESIGN §9 INV numbering belongs to
the full-suite file and PL* to the plan file; this module uses RC1-RC9
to avoid collision):

* RC1 — the four REAL artifacts reconcile clean on the set layer:
  missing=duplicates=unexpected=0, no axis/coverage/summary/meta
  findings; r01 carries exactly its 30 measured failed-row findings;
* RC2 — r01's failure layer is measured, not narrated: 30 = 15 + 15 on
  the full 3-level x 5-fold grid, all primary, one verbatim error
  string; five singles + cash + benchmark stay citable, the two combos
  are program_error; the summary layer rebuild (any-level-fails,
  r01_baseline.py:492-497) equals the recorded arm_summaries verbatim;
* RC3 — the DESIGN §6 S3 duplicate-fold counterexample at reconcile
  level: one key observed twice is REPORTED with multiplicity 2 and
  n_rows_observed=14 — never silently overwritten or deduplicated —
  while every other key still reconciles with zero false missing;
* RC4 — the DESIGN §6 S2 missing-primary-arm counterexample at
  reconcile level: deleting all 15 combo:all3 rows yields exactly those
  15 missing keys and a 'missing' arm status, while every other arm's
  measurement is preserved untouched;
* RC5 — date coverage on a SYNTHETIC calendar: within (period, fold)
  groups n_dates must be single-valued, daily_dates vectors identical,
  n_dates == len(daily_returns) per row, fold windows non-overlapping;
  a 3-day synthetic calendar is legal — reconcile never compares
  against a foreign experiment's calendar (S §6);
* RC6 — a value on an axis the plan declares inactive is an
  unexpected_axis_value finding (r01 row carrying cell; r02_matrix row
  carrying period — the INV5 anti-folding companion);
* RC7 — unplanned rows are unexpected (never silently dropped) and
  run_meta.row_count is cross-checked when run_meta is supplied;
* RC8 — the r02_periods summary layer is compared per period over its
  OWN arm domain only (F20): row-layer cash/benchmark are recorded as
  extras, never issues; a tampered/two-period-divergent rebuild is
  caught for exactly the affected period;
* RC9 — IntegrityRefusal is reserved for unauditable inputs only: no
  rows layer, a foreign-experiment report, an unknown schema version, a
  non-mapping run_meta — findings, not refusals, for everything else;
* RC10 — provenance anchoring (the comparison plan_sources' docstring
  promises): a report whose recorded provenance.config_sha256 /
  contract_sha256 does not match the plan's L1/L2 source snapshot is
  REFUSED with both measured values in the message — it was measured
  from different inputs than the plan describes, and comparing the two
  would run across mismatched calibers (the P45 lesson);
* RC11 — a MISSING anchor (either side) is disclosed as a locatable
  provenance_anchor_absent finding, never a refusal and never a silent
  skip; hex digests compare case-insensitively (the r0x writers record
  UPPERCASE config digests, _sha256_file yields lowercase — measured
  on all four artifacts), and the four REAL artifacts carry both
  anchors matching, so the real reconcile outputs gain zero findings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from ashare_model.experiment_integrity.plan import materialize_plan
from ashare_model.experiment_integrity.reconcile import (
    rebuild_failed_folds,
    reconcile_report,
    reconcile_to_json,
    row_key_of,
)
from ashare_model.experiment_integrity.types import (
    IntegrityRefusal,
    RowKey,
)

#: The frozen r1 artifact root, resolved from the package (never from
#: the process cwd) — read-only access, the WP14 red line.
_ROOT = Path(__file__).resolve().parents[1]
_R1 = _ROOT / "data" / "r1"

#: Recon-measured facts this suite anchors on (reports/wp14/_recon):
#: F2/F8/F9/F10 row counts, F4 the 30 failed rows and their verbatim
#: error string.
_VERIFIED_ROW_COUNTS = {
    "r01_slow_baseline": 125,
    "r02_matrix": 310,
    "r02_periods": 120,
    "r03_combos": 140,
}
_ARTIFACTS = {
    "r01_slow_baseline": (
        _R1 / "r01_slow_baseline" / "r01_baseline_report.json",
        _R1 / "r01_slow_baseline" / "r01_baseline_run_meta.json",
    ),
    "r02_matrix": (
        _R1 / "r02_matrix" / "r02_matrix_report.json",
        _R1 / "r02_matrix" / "r02_matrix_run_meta.json",
    ),
    "r02_periods": (
        _R1 / "r02_periods" / "r02_periods_report.json",
        _R1 / "r02_periods" / "r02_periods_run_meta.json",
    ),
    "r03_combos": (
        _R1 / "r03_combos" / "r03_combo_report.json",
        _R1 / "r03_combos" / "r03_combo_run_meta.json",
    ),
}
_VERIFIED_SLOW_SHA256 = (
    "95fac73249c0ee93f86e0f385ce25ddf5ae7e7a92bc830db0706710bc05f550b"
)
_KEY_ERROR_MOMENTUM = "KeyError: 'MOMENTUM_20'"

#: The synthetic 3-day test calendar for the mini fixtures (RC3/RC5/
#: RC6/RC7) — deliberately unlike every real artifact's 221-237-day
#: calendar: if reconcile ever compared against a foreign calendar,
#: these rows could not come out clean.
_MINI_CALENDAR = ["20210104", "20210105", "20210106"]
_SHIFTED_CALENDAR = ["20210105", "20210106", "20210107"]

_MINI_CONFIG = """\
protocol:
  domain: slow_fundamental
  tier_set: AB
  folds:
    - {train_end: "2020-12-31", test_end: "2021-12-31"}
  baseline_signals: [ROE]
"""

#: The mini r01 plan's five arms and their per-arm ladders (S §2.2):
#: 3+3+3+3+1 = 13 expected rows on one fold.
_MINI_ARMS = (
    ("baseline:ROE", ("0x", "1x", "2x")),
    ("combo:all3", ("0x", "1x", "2x")),
    ("combo:momentum_reversal", ("0x", "1x", "2x")),
    ("control:cash", ("0x", "1x", "2x")),
    ("benchmark:equal_weight", ("ref",)),
)
_MINI_CANDIDATES = tuple(candidate for candidate, _levels in _MINI_ARMS)


def _real(experiment_id):
    """(plan, report, run_meta) of one frozen artifact, read-only."""

    report_path, meta_path = _ARTIFACTS[experiment_id]
    if not report_path.is_file() or not meta_path.is_file():
        pytest.skip("optional frozen historical run archive is not shipped with source")
    plan = materialize_plan(experiment_id)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return plan, report, meta


def _mini_plan(tmp_path, config_text=_MINI_CONFIG):
    config = tmp_path / "mini.yaml"
    config.write_text(config_text, encoding="utf-8")
    return materialize_plan("r01_slow_baseline", config)


def _mini_row(
    candidate,
    cost_level,
    fold="2021-12-31",
    *,
    train_end="2020-12-31",
    failed=False,
    reason=None,
    n_dates=None,
    calendar=None,
    returns=None,
    cell=None,
):
    """One synthetic r01-shaped row: a minimal failed stub (S §2.3 — 8
    fields, no metrics) or a metric-bearing successful row whose
    calendar/returns lengths the caller controls."""

    if failed:
        return {
            "candidate": candidate,
            "formula_text": "synthetic",
            "fold_train_end": train_end,
            "fold_test_end": fold,
            "seed": None,
            "cost_level": cost_level,
            "failed": True,
            "failure_reason": reason,
        }
    dates = list(calendar or _MINI_CALENDAR)
    nd = len(dates) if n_dates is None else n_dates
    rets = list(returns) if returns is not None else [0.01] * nd
    row = {
        "candidate": candidate,
        "formula_text": "synthetic",
        "formula": "synthetic",
        "fold_train_end": train_end,
        "fold_test_end": fold,
        "seed": None,
        "cost_level": cost_level,
        "failed": False,
        "n_dates": nd,
        "daily_dates": dates,
        "daily_returns": rets,
        "benchmark_daily_returns": list(rets),
        "total_return": 0.0,
        "sharpe": 0.0,
        "excess_return": 0.0,
        "val_reward": 0.0,
    }
    if cell is not None:
        row["cell"] = cell
    return row


def _mini_rows(fold="2021-12-31", train_end="2020-12-31"):
    rows = []
    for candidate, levels in _MINI_ARMS:
        for level in levels:
            rows.append(
                _mini_row(candidate, level, fold, train_end=train_end)
            )
    return rows


def _mini_summaries(failed_folds=None):
    """A summary layer consistent with an all-successful mini report
    (or with per-arm failed_folds overrides for tamper tests)."""

    overrides = failed_folds or {}
    return {
        candidate: {"failed_folds": overrides.get(candidate, 0), "n_folds": 1}
        for candidate in _MINI_CANDIDATES
    }


def _mini_report(rows, arm_summaries=None, plan=None, **extra):
    payload = {
        "r01_report_schema_version": 3,
        "completed": True,
        "rows": rows,
        "arm_summaries": (
            _mini_summaries() if arm_summaries is None else arm_summaries
        ),
    }
    if plan is not None:
        # The provenance anchors the real reports carry (RC10/RC11):
        # the report records the config/contract sha256 it was measured
        # from, taken here from the plan's own L1/L2 snapshot so the
        # synthetic report reconciles clean like the real ones do.
        payload["provenance"] = {
            "config_path": str(plan.sources["config"]["path"]),
            "config_sha256": plan.sources["config"]["sha256"],
            "contract_path": str(plan.sources["contract"]["path"]),
            "contract_sha256": plan.sources["contract"]["sha256"],
        }
    payload.update(extra)
    return payload


# --- RC1: the four real artifacts reconcile clean on the set layer -----


@pytest.mark.parametrize("experiment_id", _VERIFIED_ROW_COUNTS)
def test_p50_rc1_real_artifacts_set_reconcile_clean(experiment_id):
    plan, report, meta = _real(experiment_id)
    recon = reconcile_report(plan, report, meta)

    assert recon.experiment_id == plan.experiment_id
    assert recon.n_rows_observed == _VERIFIED_ROW_COUNTS[experiment_id]
    assert len(recon.expected_keys) == _VERIFIED_ROW_COUNTS[experiment_id]
    assert len(recon.actual_keys) == _VERIFIED_ROW_COUNTS[experiment_id]
    assert recon.missing == ()
    assert recon.duplicates == ()
    assert recon.unexpected == ()

    codes = Counter(issue.code for issue in recon.issues)
    if experiment_id == "r01_slow_baseline":
        # F4: the ONLY findings on the slow artifact are its 30 measured
        # failed rows — no structural defect anywhere else.
        assert codes == Counter({"failed_row": 30})
    else:
        # F8/F9/F10: zero failed rows and a fully clean structural
        # reconcile for the other three artifacts.
        assert codes == Counter()
    for banned in (
        "unexpected_axis_value",
        "date_coverage_mismatch",
        "returns_length_mismatch",
        "fold_window_overlap",
        "summary_layer_mismatch",
        "meta_row_count_mismatch",
    ):
        assert codes.get(banned, 0) == 0, f"{banned} must not fire on {experiment_id}"


def test_p50_rc1_row_key_of_projects_all_five_axes():
    # §3.3: period/cell via .get (benchmark rows carry no cell field at
    # all, F8) and the three string axes read directly.
    plan, report, _meta = _real("r01_slow_baseline")
    row = report["rows"][0]
    key = row_key_of(row, plan)
    assert key == RowKey(
        period=None,
        cell=None,
        candidate=row["candidate"],
        cost_level=row["cost_level"],
        fold_test_end=row["fold_test_end"],
    )

    plan_p, report_p, _meta_p = _real("r02_periods")
    cash = next(
        r for r in report_p["rows"] if r["candidate"] == "control:cash"
    )
    bench = next(
        r for r in report_p["rows"]
        if r["candidate"] == "benchmark:equal_weight"
    )
    # F19: cash carries the reference cell on an unprefixed candidate;
    # the benchmark carries a period but never a cell.
    assert row_key_of(cash, plan_p).cell == "n20_m5000"
    assert row_key_of(bench, plan_p).cell is None
    assert row_key_of(bench, plan_p).period in (
        "every_5_days",
        "every_10_days",
    )


# --- RC2: r01's failure layer measured, arm citability, rebuild ---------


def test_p50_rc2_slow_failure_layer_is_measured_not_narrated():
    plan, report, meta = _real("r01_slow_baseline")
    recon = reconcile_report(plan, report, meta)

    # F4: 30 = 15 + 15 on the complete 3-level x 5-fold grid.
    assert len(recon.failed_rows) == 30
    by_candidate = Counter(key.candidate for key, _r in recon.failed_rows)
    assert by_candidate == {
        "combo:all3": 15,
        "combo:momentum_reversal": 15,
    }
    grid = {
        (key.cost_level, key.fold_test_end) for key, _r in recon.failed_rows
    }
    assert len(grid) == 15  # {0x,1x,2x} x 5 folds, none escaped
    # One verbatim error variant, 30/30 (F4) — sorted into the arm
    # status failure_reasons as-is, never paraphrased.
    assert {reason for _k, reason in recon.failed_rows} == {
        _KEY_ERROR_MOMENTUM
    }
    # Both combos are primary in r01 (§8) → every failed row is primary.
    assert len(recon.primary_failed_rows) == 30
    assert {k.candidate for k, _r in recon.primary_failed_rows} == {
        "combo:all3",
        "combo:momentum_reversal",
    }
    assert recon.benchmark_failed_rows == ()
    # The failed_row findings carry the error string verbatim.
    assert all(
        _KEY_ERROR_MOMENTUM in issue.detail
        for issue in recon.issues
        if issue.code == "failed_row"
    )


def test_p50_rc2_slow_arm_status_five_singles_citable():
    plan, report, meta = _real("r01_slow_baseline")
    recon = reconcile_report(plan, report, meta)
    status = {arm.candidate: arm for arm in recon.arm_status}

    citable = [
        "baseline:ROE",
        "baseline:PE_TTM",
        "baseline:REVENUE_YOY",
        "baseline:DIVIDEND_YIELD",
        "baseline:MARKET_CAP",
        "control:cash",
        "benchmark:equal_weight",
    ]
    for candidate in citable:
        arm = status[candidate]
        assert arm.status == "citable", candidate
        assert arm.n_failed_rows == 0
        assert arm.failure_reasons == ()
        if candidate == "benchmark:equal_weight":
            # The zero-cost comparison ladder is ("ref",) only: 1 x 5
            # folds (S §2.2 — per-arm ladders, never uniform).
            assert arm.n_expected_rows == arm.n_present_rows == 5
        else:
            assert arm.n_expected_rows == arm.n_present_rows == 15
    for candidate in ("combo:all3", "combo:momentum_reversal"):
        arm = status[candidate]
        assert arm.status == "program_error", candidate
        assert arm.n_failed_rows == 15
        assert arm.n_expected_rows == arm.n_present_rows == 15
        assert arm.failure_reasons == (_KEY_ERROR_MOMENTUM,)
        assert arm.is_primary is True


def test_p50_rc2_rebuild_failed_folds_matches_recorded_summary():
    # F12 / r01_baseline.py:492-497: "a fold fails for an arm when ANY
    # of its levels failed" — the rows-layer rebuild equals the frozen
    # report's arm_summaries.failed_folds verbatim, arm by arm.
    _plan, report, _meta = _real("r01_slow_baseline")
    rebuilt = rebuild_failed_folds(report["rows"])
    recorded = {
        arm: summary["failed_folds"]
        for arm, summary in report["arm_summaries"].items()
    }
    assert rebuilt == recorded
    assert rebuilt["combo:all3"] == 5
    assert rebuilt["combo:momentum_reversal"] == 5
    assert rebuilt["baseline:ROE"] == 0
    assert rebuilt["benchmark:equal_weight"] == 0

    # Unit semantics on synthetic rows: any single failed level fails
    # the fold; a fold with no failed level does not fail.
    synthetic = [
        _mini_row("baseline:A", "0x", failed=False),
        _mini_row("baseline:A", "1x", failed=True, reason="boom"),
        _mini_row("baseline:A", "0x", fold="2022-12-31", failed=False),
        _mini_row("baseline:B", "0x", failed=True, reason="boom"),
    ]
    assert rebuild_failed_folds(synthetic) == {"baseline:A": 1, "baseline:B": 1}


@pytest.mark.skipif(not _ARTIFACTS["r01_slow_baseline"][0].is_file(), reason="optional frozen historical run archive absent")
def test_p50_rc2_slow_report_digest_anchor():
    # The audited bytes (F1): every RC2 measurement above is taken on
    # exactly this file.
    report_path = _ARTIFACTS["r01_slow_baseline"][0]
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert digest == _VERIFIED_SLOW_SHA256


# --- RC3: S3 duplicate-fold counterexample (reported, never overwritten) --


def test_p50_rc3_duplicate_fold_reported_never_silently_overwritten(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    clean = reconcile_report(plan, _mini_report(rows, plan=plan))
    assert clean.n_rows_observed == 13
    assert clean.duplicates == ()
    assert clean.issues == ()

    # Duplicate one row: the SAME key observed twice (one "fold" run
    # twice) — 14 physical rows over 13 distinct keys.
    duplicated_key = row_key_of(rows[4], plan)  # combo:all3 @ 1x
    assert duplicated_key.candidate == "combo:all3"
    tampered = _mini_report(rows + [copy.deepcopy(rows[4])], plan=plan)
    recon = reconcile_report(plan, tampered)

    # RED side: the duplicate IS reported — multiplicity 2, exactly one
    # duplicate_row finding naming the key.
    assert recon.n_rows_observed == 14  # counted, not silently deduped
    assert recon.duplicates == ((duplicated_key, 2),)
    dup_issues = [
        i for i in recon.issues if i.code == "duplicate_row"
    ]
    assert len(dup_issues) == 1
    assert dup_issues[0].key == duplicated_key
    assert "2 rows share one key" in dup_issues[0].detail

    # GREEN side: no collateral damage — the other 12 keys reconcile
    # unchanged, nothing goes missing, nothing becomes unexpected, and
    # the arm-level measurement vocabulary stays unpolluted (the
    # duplicate is a report-level finding; M2 consumes it downstream).
    assert recon.missing == ()
    assert recon.unexpected == ()
    assert recon.failed_rows == ()
    assert len(recon.actual_keys) == 13
    assert duplicated_key in recon.actual_keys
    status = {arm.candidate: arm.status for arm in recon.arm_status}
    assert status == {
        "baseline:ROE": "citable",
        "combo:all3": "citable",
        "combo:momentum_reversal": "citable",
        "control:cash": "citable",
        "benchmark:equal_weight": "citable",
    }
    # A dict-key overwrite WOULD have produced 13 rows and no finding;
    # the reconciliation physically observes the 14th row.
    assert recon.n_rows_observed == len(tampered["rows"])


# --- RC4: S2 missing-primary-arm counterexample --------------------------


def test_p50_rc4_missing_primary_arm_rows_others_kept():
    plan, report, meta = _real("r01_slow_baseline")
    report = copy.deepcopy(report)
    report["rows"] = [
        row for row in report["rows"] if row["candidate"] != "combo:all3"
    ]
    assert len(report["rows"]) == 110
    recon = reconcile_report(plan, report, meta)

    # RED side: exactly the 15 deleted keys are missing — the complete
    # {0x,1x,2x} x 5-fold grid of ONE primary arm, fully locatable.
    assert len(recon.missing) == 15
    assert {key.candidate for key in recon.missing} == {"combo:all3"}
    assert {(key.cost_level, key.fold_test_end) for key in recon.missing} == {
        (level, fold)
        for level in ("0x", "1x", "2x")
        for fold in (
            "2021-12-31",
            "2022-12-31",
            "2023-12-31",
            "2024-12-31",
            "2025-12-31",
        )
    }
    assert Counter(i.code for i in recon.issues)["missing_row"] == 15
    status = {arm.candidate: arm for arm in recon.arm_status}
    assert status["combo:all3"].status == "missing"
    assert status["combo:all3"].n_present_rows == 0
    assert status["combo:all3"].n_expected_rows == 15

    # The summary layer now disagrees with the rows layer for the
    # deleted arm (recorded failed_folds=5, rebuildable 0) — an honest
    # cross-layer finding, reported as data.
    summary_issues = [
        i for i in recon.issues if i.code == "summary_layer_mismatch"
    ]
    assert len(summary_issues) == 1
    assert "combo:all3" in summary_issues[0].detail

    # GREEN side: every OTHER arm's measurement is preserved untouched.
    assert recon.duplicates == ()
    assert recon.unexpected == ()
    assert len(recon.failed_rows) == 15  # momentum_reversal, unchanged
    assert {k.candidate for k, _r in recon.failed_rows} == {
        "combo:momentum_reversal"
    }
    assert status["combo:momentum_reversal"].status == "program_error"
    assert status["combo:momentum_reversal"].n_failed_rows == 15
    for candidate in (
        "baseline:ROE",
        "baseline:PE_TTM",
        "baseline:REVENUE_YOY",
        "baseline:DIVIDEND_YIELD",
        "baseline:MARKET_CAP",
        "control:cash",
        "benchmark:equal_weight",
    ):
        assert status[candidate].status == "citable", candidate


# --- RC5: date coverage on a synthetic calendar --------------------------


def test_p50_rc5_synthetic_calendar_clean_and_foreign_calendar_never_compared(
    tmp_path,
):
    # A 3-day synthetic calendar is fully legal: reconcile checks
    # consistency WITHIN the report's (period, fold) groups only — it
    # structurally cannot compare against r02/r03's 231-232-day or
    # r01's 221-222-day calendars (S §6: cross-experiment date
    # comparison is explicitly forbidden; the 10-day horizon gap is a
    # design result, not a defect).
    plan = _mini_plan(tmp_path)
    recon = reconcile_report(plan, _mini_report(_mini_rows()))
    coverage_codes = {
        i.code for i in recon.issues
        if i.code in ("date_coverage_mismatch", "returns_length_mismatch")
    }
    assert coverage_codes == set()
    groups = recon.coverage["groups"]
    assert set(groups) == {"-|2021-12-31"}
    record = groups["-|2021-12-31"]
    assert record["n_rows"] == 13
    assert record["n_dates_observed"] == [3]
    assert record["n_dates_consistent"] is True
    assert record["daily_dates_present"] is True
    assert record["daily_dates_consistent"] is True


def test_p50_rc5_deviant_daily_dates_inside_fold_group_flagged(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    # One arm trades a shifted 3-day calendar inside the same fold.
    rows[3]["daily_dates"] = list(_SHIFTED_CALENDAR)
    recon = reconcile_report(plan, _mini_report(rows))

    mismatches = [
        i for i in recon.issues if i.code == "date_coverage_mismatch"
    ]
    assert len(mismatches) == 1
    assert "distinct daily_dates vectors" in mismatches[0].detail
    assert "2021-12-31" in mismatches[0].detail  # locatable group
    assert (
        recon.coverage["groups"]["-|2021-12-31"]["daily_dates_consistent"]
        is False
    )
    # n_dates itself still agrees (all rows say 3): the calendar-vector
    # defect is named precisely, not conflated with a count defect.
    assert (
        recon.coverage["groups"]["-|2021-12-31"]["n_dates_consistent"]
        is True
    )
    # The row still reconciles on the key layer — this is a coverage
    # finding, not a missing/duplicate/unexpected row.
    assert recon.missing == () and recon.duplicates == ()
    assert recon.unexpected == ()


def test_p50_rc5_disagreeing_n_dates_inside_fold_group_flagged(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    # One row CLAIMS 4 dates with a matching 4-long returns vector while
    # keeping the shared 3-day calendar vector: an isolated within-group
    # n_dates disagreement — no calendar-vector defect, no per-row
    # length lie — and the finding must name exactly that.
    rows[1]["n_dates"] = 4
    rows[1]["daily_returns"] = [0.01, 0.01, 0.01, 0.01]
    recon = reconcile_report(plan, _mini_report(rows))

    mismatches = [
        i for i in recon.issues if i.code == "date_coverage_mismatch"
    ]
    assert len(mismatches) == 1
    assert "disagree on n_dates" in mismatches[0].detail
    assert "[3, 4]" in mismatches[0].detail
    assert (
        recon.coverage["groups"]["-|2021-12-31"]["n_dates_observed"]
        == [3, 4]
    )
    assert [
        i for i in recon.issues if i.code == "returns_length_mismatch"
    ] == []


def test_p50_rc5_returns_length_mismatch_flagged_with_row_key(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    benchmark_row = rows[12]
    assert benchmark_row["candidate"] == "benchmark:equal_weight"
    benchmark_row["daily_returns"] = [0.01, 0.02]  # n_dates says 3
    recon = reconcile_report(plan, _mini_report(rows))

    length_issues = [
        i for i in recon.issues if i.code == "returns_length_mismatch"
    ]
    assert len(length_issues) == 1
    assert length_issues[0].key == row_key_of(benchmark_row, plan)
    assert "n_dates=3" in length_issues[0].detail
    assert "len(daily_returns)=2" in length_issues[0].detail
    # The calendar layer itself stays consistent: this is a per-row
    # vector-length lie, precisely separated from a calendar defect.
    assert [
        i for i in recon.issues if i.code == "date_coverage_mismatch"
    ] == []


def test_p50_rc5_overlapping_fold_windows_flagged(tmp_path):
    config = """\
protocol:
  domain: slow_fundamental
  tier_set: AB
  folds:
    - {train_end: "2020-12-31", test_end: "2021-12-31"}
    - {train_end: "2021-06-30", test_end: "2022-12-31"}
  baseline_signals: [ROE]
"""
    plan = _mini_plan(tmp_path, config)
    assert plan.expected_row_count == 26
    rows = _mini_rows() + _mini_rows(
        fold="2022-12-31", train_end="2021-06-30"
    )
    recon = reconcile_report(
        plan,
        _mini_report(
            rows,
            arm_summaries={
                candidate: {"failed_folds": 0, "n_folds": 2}
                for candidate in _MINI_CANDIDATES
            },
        ),
    )
    # The second fold's test window starts (2021-06-30) before the
    # first fold's window ends (2021-12-31) — observed from the rows.
    overlaps = [
        i for i in recon.issues if i.code == "fold_window_overlap"
    ]
    assert len(overlaps) == 1
    assert "2021-06-30" in overlaps[0].detail
    assert "2021-12-31" in overlaps[0].detail
    assert "overlap" in overlaps[0].detail
    # Everything else on the overlapping-window report is still clean.
    assert recon.missing == () and recon.duplicates == ()
    assert recon.unexpected == ()
    assert recon.n_rows_observed == 26


# --- RC6: inactive-axis values are findings (anti-folding companion) -----


def test_p50_rc6_r01_row_carrying_cell_is_an_unexpected_axis_value(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    foreign = _mini_row("baseline:FOREIGN", "0x", cell="n20_m5000")
    rows.append(foreign)
    recon = reconcile_report(plan, _mini_report(rows))

    axis_issues = [
        i for i in recon.issues if i.code == "unexpected_axis_value"
    ]
    assert len(axis_issues) == 1
    assert "'cell'" in axis_issues[0].detail
    assert "n20_m5000" in axis_issues[0].detail
    assert axis_issues[0].key == row_key_of(foreign, plan)
    # The same row is ALSO unplanned (r01 declares no cell axis at all):
    # both findings fire together — the axis defect never masks the
    # set defect, and neither masks the other.
    assert row_key_of(foreign, plan) in recon.unexpected
    assert recon.missing == ()
    assert recon.duplicates == ()


def test_p50_rc6_r02_matrix_row_carrying_period_is_an_unexpected_axis_value():
    # F9's anti-folding companion: r02_matrix's key axes have no period;
    # a row smuggling one in is flagged.  (r02_periods rows DO carry
    # period legally — that axis is declared active there.)
    plan, report, meta = _real("r02_matrix")
    report = copy.deepcopy(report)
    smuggled = copy.deepcopy(report["rows"][0])
    smuggled["period"] = "every_5_days"
    report["rows"].append(smuggled)
    recon = reconcile_report(plan, report, meta)

    axis_issues = [
        i for i in recon.issues if i.code == "unexpected_axis_value"
    ]
    assert len(axis_issues) == 1
    assert "'period'" in axis_issues[0].detail
    assert "every_5_days" in axis_issues[0].detail
    assert row_key_of(smuggled, plan) in recon.unexpected
    assert recon.missing == ()
    assert recon.duplicates == ()
    # The 310 genuine rows are untouched by the smuggled 311th.
    assert recon.n_rows_observed == 311
    assert len(recon.actual_keys) == 311


# --- RC7: unplanned rows and the run_meta row-count cross-check ----------


def test_p50_rc7_unplanned_row_is_unexpected_never_dropped(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    foreign = _mini_row("baseline:FOREIGN", "0x")
    rows.append(foreign)
    recon = reconcile_report(plan, _mini_report(rows))

    assert recon.unexpected == (row_key_of(foreign, plan),)
    assert recon.missing == ()
    assert recon.n_rows_observed == 14  # observed, not silently dropped
    unexpected_issues = [
        i for i in recon.issues if i.code == "unexpected_row"
    ]
    assert len(unexpected_issues) == 1
    assert "baseline:FOREIGN" in unexpected_issues[0].detail
    # Foreign arms never enter the plan's arm-status layer.
    assert "baseline:FOREIGN" not in {
        arm.candidate for arm in recon.arm_status
    }


def test_p50_rc7_meta_row_count_cross_checked_when_supplied(tmp_path):
    plan = _mini_plan(tmp_path)
    report = _mini_report(_mini_rows(), plan=plan)

    # No run_meta → no meta finding (the check is opt-in).
    assert reconcile_report(plan, report).issues == ()
    # Correct run_meta → still no finding.
    assert reconcile_report(plan, report, {"row_count": 13}).issues == ()
    # Wrong run_meta → one locatable meta_row_count_mismatch finding.
    recon = reconcile_report(plan, report, {"row_count": 12})
    meta_issues = [
        i for i in recon.issues if i.code == "meta_row_count_mismatch"
    ]
    assert len(meta_issues) == 1
    assert "row_count=12" in meta_issues[0].detail
    assert "len(rows)=13" in meta_issues[0].detail
    # An absent row_count key is a visible mismatch, never a silent pass.
    recon = reconcile_report(plan, report, {"state": "RETIRED"})
    assert [
        i for i in recon.issues if i.code == "meta_row_count_mismatch"
    ]


def test_p50_rc7_reconcile_to_json_is_jsonable_and_countable(tmp_path):
    plan = _mini_plan(tmp_path)
    rows = _mini_rows()
    dup = copy.deepcopy(rows[4])
    rows[3].update(failed=True, failure_reason="boom")  # combo:all3 @ 0x
    recon = reconcile_report(plan, _mini_report(rows + [dup], plan=plan))

    payload = reconcile_to_json(recon)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    assert json.loads(encoded) == payload
    assert payload["experiment_id"] == "r01_slow_baseline"
    assert payload["n_rows_observed"] == 14
    assert payload["n_missing"] == len(recon.missing) == 0
    assert payload["n_duplicates"] == 1
    assert payload["n_unexpected"] == 0
    assert payload["n_failed_rows"] == 1
    assert payload["failed_rows"][0]["failure_reason"] == "boom"
    assert payload["duplicates"][0]["multiplicity"] == 2
    codes = Counter(i["code"] for i in payload["issues"])
    assert codes == Counter(
        {"duplicate_row": 1, "failed_row": 1, "summary_layer_mismatch": 1}
    )
    # ^ the tampered failed row breaks the all-zero summary rebuild for
    # combo:all3 (recorded 0, rebuilt 1) — consistent with RC8 semantics.

    # The real slow artifact's payload carries the measured facts.
    plan_real, report_real, meta_real = _real("r01_slow_baseline")
    real_payload = reconcile_to_json(
        reconcile_report(plan_real, report_real, meta_real)
    )
    assert real_payload["n_rows_observed"] == 125
    assert real_payload["n_failed_rows"] == 30
    assert real_payload["n_primary_failed_rows"] == 30
    assert real_payload["n_benchmark_failed_rows"] == 0
    assert {f["failure_reason"] for f in real_payload["failed_rows"]} == {
        _KEY_ERROR_MOMENTUM
    }
    citable = {
        a["candidate"] for a in real_payload["arm_status"]
        if a["status"] == "citable"
    }
    assert citable == {
        "baseline:ROE",
        "baseline:PE_TTM",
        "baseline:REVENUE_YOY",
        "baseline:DIVIDEND_YIELD",
        "baseline:MARKET_CAP",
        "control:cash",
        "benchmark:equal_weight",
    }
    assert hashlib.sha256(
        _ARTIFACTS["r01_slow_baseline"][0].read_bytes()
    ).hexdigest() == _VERIFIED_SLOW_SHA256


# --- RC8: the r02_periods summary layer over its own arm domain ----------


def test_p50_rc8_r02_periods_summary_domain_and_period_prefilter():
    plan, report, meta = _real("r02_periods")
    recon = reconcile_report(plan, report, meta)

    # F20: the summary layer's arm domain is ONLY the 5 reference-cell
    # signal arms per period — the row layer's cash and benchmark are
    # recorded as extras in coverage, NEVER as issues.
    assert [
        i for i in recon.issues if i.code == "summary_layer_mismatch"
    ] == []
    extras = recon.coverage["summary_layer_extra_arms"]
    assert sorted(extras) == sorted(
        [
            "period='every_5_days':benchmark:equal_weight",
            "period='every_5_days':control:cash",
            "period='every_10_days':benchmark:equal_weight",
            "period='every_10_days':control:cash",
        ]
    )
    # The per-period rebuild matches the recorded failed_folds (0) for
    # every DOMAIN arm (the row layer's cash/benchmark rebuild values
    # are the extras above, outside this comparison — F20).
    for period, block in report["period_blocks"].items():
        rows_p = [r for r in report["rows"] if r.get("period") == period]
        rebuilt = rebuild_failed_folds(rows_p)
        assert {
            arm: summary["failed_folds"]
            for arm, summary in block["arms"].items()
        } == {arm: rebuilt[arm] for arm in block["arms"]}

    # Tamper ONE period's arm: the mismatch fires for exactly that
    # period — proof the rows were period-prefiltered before the
    # rebuild (without the prefilter, both periods' summaries would
    # disagree and two findings would fire).
    tampered = copy.deepcopy(report)
    arm_name = sorted(tampered["period_blocks"]["every_5_days"]["arms"])[0]
    tampered["period_blocks"]["every_5_days"]["arms"][arm_name][
        "failed_folds"
    ] = 1
    recon2 = reconcile_report(plan, tampered, meta)
    summary_issues = [
        i for i in recon2.issues if i.code == "summary_layer_mismatch"
    ]
    assert len(summary_issues) == 1
    assert "every_5_days" in summary_issues[0].detail
    assert arm_name in summary_issues[0].detail
    assert "recorded=1" in summary_issues[0].detail
    assert "rebuilt=0" in summary_issues[0].detail


def test_p50_rc8_declared_summary_layer_absent_is_fail_visible(tmp_path):
    # A report whose DECLARED summary layer is missing cannot be
    # cross-checked: that is a summary_layer_mismatch FINDING (the rows
    # still reconcile), not a refusal — findings are data.
    plan = _mini_plan(tmp_path)
    report = _mini_report(_mini_rows())
    del report["arm_summaries"]
    recon = reconcile_report(plan, report)
    summary_issues = [
        i for i in recon.issues if i.code == "summary_layer_mismatch"
    ]
    assert len(summary_issues) == 1
    assert "arm_summaries" in summary_issues[0].detail
    assert recon.missing == () and recon.duplicates == ()
    assert recon.unexpected == ()


# --- RC9: refusals reserved for unauditable inputs only ------------------


def test_p50_rc9_refusals_for_unauditable_inputs(tmp_path):
    plan = _mini_plan(tmp_path)

    # No rows layer at all → nothing to reconcile against the plan.
    with pytest.raises(IntegrityRefusal, match=r"locator=rows"):
        reconcile_report(
            plan,
            {"r01_report_schema_version": 3, "completed": True},
        )

    # A foreign experiment's report under this plan (r03 family key).
    foreign = _mini_report(_mini_rows())
    del foreign["r01_report_schema_version"]
    foreign["r03_report_schema_version"] = 1
    with pytest.raises(IntegrityRefusal, match=r"experiment mismatch"):
        reconcile_report(plan, foreign)

    # Right family, unknown schema version value.
    future = _mini_report(_mini_rows())
    future["r01_report_schema_version"] = 99
    with pytest.raises(IntegrityRefusal, match=r"unknown schema version"):
        reconcile_report(plan, future)

    # A row that is not a mapping cannot be keyed at all.
    broken = _mini_report(_mini_rows())
    broken["rows"].append("not-a-mapping")
    with pytest.raises(IntegrityRefusal, match=r"rows\[13\]"):
        reconcile_report(plan, broken)

    # run_meta garbage refuses; run_meta absent never does.
    with pytest.raises(IntegrityRefusal, match=r"locator=run_meta"):
        reconcile_report(plan, _mini_report(_mini_rows()), run_meta=[1, 2])


def test_p50_rc9_cross_experiment_report_refused_on_real_artifacts():
    # The r03 artifact reconciles under its own plan but is REFUSED
    # under the r01 plan (identity, not a findings dump).
    r01_plan = materialize_plan("r01_slow_baseline")
    _r03_plan, r03_report, _meta = _real("r03_combos")
    with pytest.raises(IntegrityRefusal, match=r"r01_report_schema_version"):
        reconcile_report(r01_plan, r03_report)
    # And symmetrically the r01 artifact under the r03 plan.
    r03_plan = materialize_plan("r03_combos")
    _plan, r01_report, _meta = _real("r01_slow_baseline")
    with pytest.raises(IntegrityRefusal, match=r"r03_report_schema_version"):
        reconcile_report(r03_plan, r01_report)


# --- RC10: provenance anchor mismatch refuses with both values ------------


def test_p50_rc10_provenance_anchor_mismatch_refuses_with_both_values():
    # plan.plan_sources' docstring promises "a later reconcile compares
    # them against the report's recorded provenance" — RC10/RC11 are
    # that comparison.  A report whose recorded config/contract sha256
    # does not match the plan's L1/L2 snapshot was measured from
    # DIFFERENT inputs than the plan describes; reconciling the two
    # would compare across mismatched calibers (the P45 lesson), so it
    # is an IntegrityRefusal — identity, not a findings dump — with
    # BOTH measured values and the locator in the message.
    plan, report, meta = _real("r01_slow_baseline")

    tampered = copy.deepcopy(report)
    tampered["provenance"]["config_sha256"] = "0" * 64
    with pytest.raises(IntegrityRefusal) as excinfo:
        reconcile_report(plan, tampered, meta)
    message = str(excinfo.value)
    assert "provenance.config_sha256" in message
    assert plan.sources["config"]["sha256"] in message  # plan side value
    assert "0" * 64 in message  # report side value
    assert "r01_slow_baseline" in message

    tampered_contract = copy.deepcopy(report)
    tampered_contract["provenance"]["contract_sha256"] = "f" * 64
    with pytest.raises(IntegrityRefusal, match=r"contract_sha256"):
        reconcile_report(plan, tampered_contract, meta)


# --- RC11: missing anchors are disclosed findings, never silent -----------


def test_p50_rc11_missing_provenance_anchors_are_disclosed_findings():
    # The four REAL artifacts carry both anchors and they match the
    # materialized plans (8/8 measured) — zero new findings on the real
    # reconcile outputs; the anchoring comparison is additive only for
    # inputs that actually drift.
    for experiment_id in _VERIFIED_ROW_COUNTS:
        plan, report, meta = _real(experiment_id)
        recon = reconcile_report(plan, report, meta)
        assert not [
            i for i in recon.issues if i.code == "provenance_anchor_absent"
        ], experiment_id

    plan, report, meta = _real("r01_slow_baseline")
    stripped = copy.deepcopy(report)
    del stripped["provenance"]
    recon = reconcile_report(plan, stripped, meta)
    absent = [i for i in recon.issues if i.code == "provenance_anchor_absent"]
    assert len(absent) == 2  # config + contract, one finding each
    assert any("config" in i.detail for i in absent)
    assert any("contract" in i.detail for i in absent)
    assert all(i.key is None for i in absent)  # report-level, not row-level
    # Disclosure, not refusal: the rows layer still reconciles in full.
    assert recon.n_rows_observed == 125
    assert recon.missing == () and recon.duplicates == ()
    assert recon.unexpected == ()

    partial = copy.deepcopy(report)
    del partial["provenance"]["contract_sha256"]
    recon_partial = reconcile_report(plan, partial, meta)
    assert len(
        [i for i in recon_partial.issues if i.code == "provenance_anchor_absent"]
    ) == 1

    # Hex digests compare case-insensitively: the r0x writers record
    # UPPERCASE config digests while _sha256_file yields lowercase
    # (measured on all four artifacts) — an uppercased copy of the
    # plan-side digest is a MATCH, not a refusal.
    upper = copy.deepcopy(report)
    upper["provenance"]["config_sha256"] = plan.sources["config"]["sha256"].upper()
    assert not [
        i
        for i in reconcile_report(plan, upper, meta).issues
        if i.code == "provenance_anchor_absent"
    ]
