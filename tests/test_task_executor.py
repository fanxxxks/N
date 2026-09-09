"""E3 contract tests: the reliable long-task executor (t5).

The executor runs every (seed, backend, track, fold) row of a plan in an
INDEPENDENT subprocess, reports heartbeat + resource metrics (CPU / RSS /
wall), binds every attempt to the run's RunStore ledger (spec_id/run_id
identity, hash-chained, append-only), writes row results atomically
(tmp + os.replace — an injected write failure leaves zero half rows),
and on restart reconciles the ledger: rows with a terminal record are
never re-executed (a failed trial stays in the chain; redoing it is a
rerun, which means a NEW run id / output directory), while an interrupted
attempt (open trial, no valid result file) is closed as failed and the
row is re-executed — only unfinished rows are filled.  An interrupted
attempt whose worker did leave a valid result file is adopted from that
result without re-running the row.

The row command is opaque to the executor: ``command_builder(row) ->
argv``.  Exit code 0 means the row's work succeeded; anything else is a
recorded measurement (failed), never silently dropped.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ashare_data.config import BacktestConfig
from ashare_model import task_executor as te
from ashare_model.ledger import ExperimentLedger
from ashare_model.run_store import RunLockedError, RunStore
from ashare_model.runspec import WindowSpec, build_runspec
from ashare_portfolio.execution_spec import portfolio_config_provenance


def _spec():
    return build_runspec(
        dataset_id="ds-executor-0001",
        data_cutoff="20241231",
        frequency="daily",
        target="forward_return",
        horizon=1,
        requested_budget=2000,
        n_folds=2,
        dev_window=WindowSpec(start="20150101", end="20231231"),
        holdout_window=WindowSpec(start="20240101", end="20241231"),
        preregistered_contract_hash="a" * 64,
        resolved_thresholds={"factor_min_coverage": 0.9, "min_eligible": 100.0},
        portfolio_config=portfolio_config_provenance(BacktestConfig()),
        require_clean_tree=False,
    )


def _rows(*triples):
    """TaskRows from (seed, backend, fold) triples with track='research'."""

    return [
        te.TaskRow(seed=seed, backend=backend, track="research", fold=fold)
        for seed, backend, fold in triples
    ]


def _ok_cmd(marker: Path, label: str) -> list[str]:
    code = (
        "import pathlib; "
        f"p = pathlib.Path({str(marker)!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "lines = p.read_text('utf-8').splitlines() if p.exists() else []; "
        f"lines.append({label!r}); "
        "p.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')"
    )
    return [sys.executable, "-c", code]


def _fail_cmd(code: int = 3) -> list[str]:
    return [sys.executable, "-c", f"raise SystemExit({code})"]


def _run_dir(store: RunStore, spec, run_id: str) -> Path:
    return store.runs_dir / spec.spec_id / run_id


def _read_ledger(store: RunStore, spec, run_id: str) -> ExperimentLedger:
    """Fresh reader over the run's bound ledger (chain verified on load)."""

    return ExperimentLedger(
        _run_dir(store, spec, run_id) / "trials.jsonl",
        run_id=run_id,
        spec_id=spec.spec_id,
    )


def _attempts_by_row(ledger: ExperimentLedger) -> dict[str, list[dict]]:
    """Per row_id: ordered attempts [{trial_id, status}]."""

    opens: dict[str, str] = {}
    attempts: dict[str, list[dict]] = {}
    for entry in ledger.iter_entries():
        row_id = str((entry.payload or {}).get("row_id"))
        if entry.status == "running":
            opens[entry.trial_id] = row_id
            attempts.setdefault(row_id, []).append(
                {"trial_id": entry.trial_id, "status": "running"}
            )
        else:
            assert opens.get(entry.trial_id) == row_id, (
                "terminal entry without a matching open trial"
            )
            attempts[row_id][-1]["status"] = entry.status
    return attempts


def _worker_result_payload(row, status: str) -> dict:
    return {
        "result_schema_version": te.TASK_EXECUTOR_SCHEMA_VERSION,
        "row_id": row.row_id,
        "status": status,
        "exit_code": 0 if status == "succeeded" else 1,
        "wall_seconds": 0.05,
        "cpu_seconds": 0.01,
        "peak_rss_mb": 20.0,
        "error": None if status == "succeeded" else "boom",
        "started_at": "2026-09-04T00:00:00+00:00",
        "finished_at": "2026-09-04T00:00:01+00:00",
        "author": "worker",
    }


# ------------------------------------------------------------ plan validation


def test_plan_rejects_empty_duplicate_rows_and_slug_collision():
    with pytest.raises(ValueError):
        te.build_plan([])
    with pytest.raises(ValueError):
        te.build_plan(_rows((7, "gp", "0"), (7, "gp", "0")))
    # Distinct rows whose file slugs collide are refused fail-closed.
    rows = [
        te.TaskRow(seed=7, backend="gp", track="a_b", fold="0"),
        te.TaskRow(seed=7, backend="gp", track="a", fold="b:0"),
    ]
    assert rows[0].row_id != rows[1].row_id
    assert te.row_slug(rows[0]) == te.row_slug(rows[1])
    with pytest.raises(ValueError):
        te.build_plan(rows)
    plan = te.build_plan(_rows((7, "gp", "0"), (8, "tpe", "1")))
    assert [row.row_id for row in plan] == [
        "7:gp:research:0",
        "8:tpe:research:1",
    ]


# ------------------------------------------------- independent subprocess rows


def test_rows_run_in_independent_subprocesses_with_heartbeat_and_metrics(tmp_path):
    marker = tmp_path / "invocations.log"
    rows = _rows((7, "gp", "0"), (8, "tpe", "1"))
    store = RunStore(tmp_path / "store")
    executor = te.TaskExecutor(
        store=store,
        spec=_spec(),
        rows=rows,
        command_builder=lambda row: _ok_cmd(marker, row.row_id),
        run_id="run-subproc",
        heartbeat_interval_s=1.0,
    )
    payload = executor.execute()
    assert payload["run_id"] == "run-subproc"
    assert payload["status"] == "completed"
    assert set(payload["rows"]) == {row.row_id for row in rows}
    assert all(info["status"] == "succeeded" for info in payload["rows"].values())

    # Each row ran in its own worker subprocess (independent PIDs) with
    # its own nested command process, and produced observable metrics.
    worker_pids = set()
    child_pids = set()
    for row in rows:
        result = json.loads(
            (executor.rows_dir / f"{te.row_slug(row)}.result.json").read_text("utf-8")
        )
        assert result["result_schema_version"] == te.TASK_EXECUTOR_SCHEMA_VERSION
        assert result["status"] == "succeeded"
        assert result["exit_code"] == 0
        assert result["wall_seconds"] > 0
        assert result["cpu_seconds"] >= 0
        assert result["peak_rss_mb"] is not None and result["peak_rss_mb"] > 0
        worker_pids.add(result["worker_pid"])
        child_pids.add(result["child_pid"])
        assert result["child_pid"] != result["worker_pid"]
        heartbeat = json.loads(
            (
                executor.rows_dir / f"{te.row_slug(row)}.heartbeat.json"
            ).read_text("utf-8")
        )
        assert (
            heartbeat["heartbeat_schema_version"] == te.TASK_EXECUTOR_SCHEMA_VERSION
        )
        assert heartbeat["row_id"] == row.row_id
        assert heartbeat["pid"] == result["worker_pid"]
        assert heartbeat["rss_mb"] > 0
        assert heartbeat["wall_seconds"] > 0
        assert "cpu_percent" in heartbeat
    assert len(worker_pids) == len(rows)
    assert len(child_pids) == len(rows)

    # Ledger binding: one succeeded attempt per row, identity bound.
    ledger = _read_ledger(store, _spec(), "run-subproc")
    attempts = _attempts_by_row(ledger)
    assert {
        row.row_id: [a["status"] for a in att] for row, att in zip(rows, attempts.values())
    } == {row.row_id: ["succeeded"] for row in rows}
    # Marker: each row's command invoked exactly once.
    lines = marker.read_text("utf-8").splitlines()
    assert sorted(lines) == sorted(row.row_id for row in rows)


# ------------------------------------------------------------ atomic row write


def test_row_result_write_injected_failure_leaves_zero_half_rows(
    tmp_path, monkeypatch
):
    target = tmp_path / "row.result.json"
    payload = {"result_schema_version": 1, "status": "succeeded"}

    te.write_row_result(target, payload)
    first = target.read_text(encoding="utf-8")

    # Inject the failure exactly at the rename instant (the half-row risk
    # point): the tmp file may survive, the final row file must never end
    # up half-written, and a pre-existing complete result stays intact.
    def boom(src, dst):
        raise OSError("injected replace failure")

    with monkeypatch.context() as ctx:
        ctx.setattr(te.os, "replace", boom)
        with pytest.raises(OSError):
            te.write_row_result(
                target, {"result_schema_version": 1, "status": "failed"}
            )
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert target.read_text(encoding="utf-8") == first

    fresh = tmp_path / "other.result.json"
    with monkeypatch.context() as ctx:
        ctx.setattr(te.os, "replace", boom)
        with pytest.raises(OSError):
            te.write_row_result(fresh, payload)
    assert not fresh.exists()  # zero half rows: no final file appears
    leftovers = list(tmp_path.glob("*.tmp.json"))
    # The tmp siblings hold the complete payload — the atomic-rename
    # boundary means a reader of the final path sees all-or-nothing.
    failed_payload = {"result_schema_version": 1, "status": "failed"}
    assert leftovers, "expected the injected failures to leave tmp files"
    assert all(
        json.loads(p.read_text("utf-8")) in (payload, failed_payload)
        for p in leftovers
    )


# --------------------------------------------------------- heartbeat timeout


def test_supervisor_detects_heartbeat_timeout_and_continues(tmp_path):
    marker = tmp_path / "invocations.log"
    rows = _rows((7, "gp", "0"), (8, "tpe", "1"))
    store = RunStore(tmp_path / "store")
    executor = te.TaskExecutor(
        store=store,
        spec=_spec(),
        rows=rows,
        command_builder=lambda row: (
            [sys.executable, "-c", "import time; time.sleep(120)"]
            if row.seed == 7
            else _ok_cmd(marker, row.row_id)
        ),
        run_id="run-timeout",
        heartbeat_interval_s=60.0,  # only the initial beat arrives
        # Must stay comfortably above worker process-spawn time (the age
        # clock starts at the attempt start until the first beat lands).
        heartbeat_timeout_s=8.0,
    )
    started = time.monotonic()
    payload = executor.execute()
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"timeout detection hung the executor ({elapsed:.1f}s)"

    ledger = _read_ledger(store, _spec(), "run-timeout")
    attempts = _attempts_by_row(ledger)
    assert [a["status"] for a in attempts["7:gp:research:0"]] == ["failed"]
    assert [a["status"] for a in attempts["8:tpe:research:1"]] == ["succeeded"]
    timeout_error = next(
        entry.error
        for entry in ledger.iter_entries()
        if entry.status == "failed" and entry.error
        if "heartbeat timeout" in entry.error
    )
    assert "7:gp:research:0" in timeout_error
    # The supervisor wrote the abnormal terminal row record itself.
    result = json.loads(
        (
            executor.rows_dir / "7_gp_research_0.result.json"
        ).read_text("utf-8")
    )
    assert result["status"] == "failed"
    assert result["author"] == "supervisor"
    assert "heartbeat timeout" in result["error"]
    # The timed-out row was killed; the next row still completed.
    assert marker.read_text("utf-8").splitlines() == ["8:tpe:research:1"]


# ------------------------------------------------- crash isolation (one row)


def test_command_crash_is_recorded_and_does_not_affect_other_rows(tmp_path):
    marker = tmp_path / "invocations.log"
    rows = _rows((7, "gp", "0"), (8, "tpe", "1"))
    store = RunStore(tmp_path / "store")
    executor = te.TaskExecutor(
        store=store,
        spec=_spec(),
        rows=rows,
        command_builder=lambda row: (
            _fail_cmd(3) if row.seed == 7 else _ok_cmd(marker, row.row_id)
        ),
        run_id="run-crash",
    )
    payload = executor.execute()
    assert payload["rows"]["7:gp:research:0"]["status"] == "failed"
    assert payload["rows"]["8:tpe:research:1"]["status"] == "succeeded"

    ledger = _read_ledger(store, _spec(), "run-crash")
    attempts = _attempts_by_row(ledger)
    assert [a["status"] for a in attempts["7:gp:research:0"]] == ["failed"]
    assert [a["status"] for a in attempts["8:tpe:research:1"]] == ["succeeded"]
    # The crash is a recorded measurement with the failing exit code.
    crash = next(
        entry
        for entry in ledger.iter_entries()
        if entry.status == "failed" and entry.error
    )
    assert "7:gp:research:0" in crash.error
    assert crash.metrics["exit_code"] == 3
    # The surviving row completed exactly once.
    assert marker.read_text("utf-8").splitlines() == ["8:tpe:research:1"]


def test_worker_dying_without_result_is_recorded_by_supervisor(tmp_path):
    """A worker that dies before writing its result (hard kill / segfault)
    is closed fail-closed by the supervisor: terminal failed row record +
    ledger entry naming the row — other rows are unaffected."""

    marker = tmp_path / "invocations.log"
    rows = _rows((7, "gp", "0"), (8, "tpe", "1"))
    store = RunStore(tmp_path / "store")
    executor = te.TaskExecutor(
        store=store,
        spec=_spec(),
        rows=rows,
        command_builder=lambda row: _ok_cmd(marker, row.row_id),
        run_id="run-noworker",
    )
    real_spawn = executor._spawn_worker

    def spawn(job_path):
        # Seam substitution for row 7 only: a real subprocess that exits
        # immediately without writing any result file.
        if "7_gp_research_0" in Path(str(job_path)).name:
            return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
        return real_spawn(job_path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(executor, "_spawn_worker", spawn)
    try:
        payload = executor.execute()
    finally:
        monkeypatch.undo()
    assert payload["rows"]["7:gp:research:0"]["status"] == "failed"
    assert payload["rows"]["8:tpe:research:1"]["status"] == "succeeded"

    ledger = _read_ledger(store, _spec(), "run-noworker")
    crash_error = next(
        entry.error
        for entry in ledger.iter_entries()
        if entry.status == "failed" and entry.error
        if "without a valid result" in entry.error
    )
    assert "7:gp:research:0" in crash_error
    result = json.loads(
        (
            executor.rows_dir / "7_gp_research_0.result.json"
        ).read_text("utf-8")
    )
    assert result["author"] == "supervisor"
    assert result["exit_code"] == 7
    assert marker.read_text("utf-8").splitlines() == ["8:tpe:research:1"]


# ------------------------------------------------------- resume / reconciliation


def _open_interrupted_attempt(store: RunStore, spec, run_id: str, row) -> str:
    """Recreate the exact on-disk state a hard-killed supervisor leaves:
    an open (running) trial for the row, never closed."""

    handle = store.attach_run(spec, run_id)
    try:
        return handle.record_trial(
            algorithm=f"backend:{row.backend}",
            candidate=row.row_id,
            seed=row.seed,
            payload={
                "row_id": row.row_id,
                "backend": row.backend,
                "track": row.track,
                "fold": row.fold,
                "attempt": 1,
            },
        )
    finally:
        handle.close()


def test_resume_fills_only_unfinished_rows_and_preserves_failed_trial(tmp_path):
    marker = tmp_path / "invocations.log"
    row_a, row_b, row_c = _rows((7, "gp", "0"), (8, "tpe", "1"), (9, "rl", "2"))
    store = RunStore(tmp_path / "store")
    spec = _spec()
    run_id = "run-resume"
    ledger_path = _run_dir(store, spec, run_id) / "trials.jsonl"

    # Phase A (real supervisor subprocess): row_a succeeds, row_b fails
    # (terminal), row_c is mid-flight when the whole supervisor tree is
    # killed — the exact on-disk state a hard crash leaves.
    phase_a = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, r"{Path(__file__).resolve().parent}")
from ashare_model.run_store import RunStore
from ashare_model import task_executor as te
from test_task_executor import _spec, _rows, _ok_cmd, _fail_cmd
cfg = json.loads({json.dumps(json.dumps({"store": str(tmp_path / "store"), "marker": str(marker)}))})
spec = _spec()
rows = _rows((7, "gp", "0"), (8, "tpe", "1"), (9, "rl", "2"))
def builder(row):
    if row.seed == 7:
        return _ok_cmd(Path(cfg["marker"]), row.row_id)
    if row.seed == 8:
        return _fail_cmd(3)
    return [sys.executable, "-c", "import time; time.sleep(60)"]
executor = te.TaskExecutor(
    store=RunStore(cfg["store"]),
    spec=spec,
    rows=rows,
    command_builder=builder,
    run_id="run-resume",
    heartbeat_interval_s=1.0,
)
executor.execute()
"""
    proc_a = subprocess.Popen(
        [sys.executable, "-c", phase_a],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 90.0
    saw_open_c = False
    while time.monotonic() < deadline:
        if proc_a.poll() is not None:
            break
        if ledger_path.exists():
            entries = [
                json.loads(line)
                for line in ledger_path.read_text("utf-8").splitlines()
                if line.strip()
            ]
            if len(entries) >= 5 and entries[-1]["status"] == "running":
                assert (
                    entries[-1]["payload"]["row_id"] == row_c.row_id
                ), entries[-1]
                saw_open_c = True
                break
        time.sleep(0.2)
    assert saw_open_c, "phase A never reached row_c's open trial"
    te._kill_tree(proc_a.pid, grace_s=5.0)
    proc_a.wait(timeout=30)
    assert proc_a.poll() is not None

    # The dead supervisor left: open trial for row_c, no result for it,
    # and the stale single-writer lock — recovered explicitly (RunStore
    # contract: stale-lock clearing is never automatic).
    assert not (
        _run_dir(store, spec, run_id) / f"executor_rows/{te.row_slug(row_c)}.result.json"
    ).exists()
    assert store.clear_stale_lock(spec.spec_id, run_id) is True

    # Phase B (in-process resume): row_a must not re-run (terminal
    # success), row_b must not re-run (failed trial preserved — redoing
    # it is a rerun, i.e. a new run id/output dir), row_c's interrupted
    # attempt is closed as failed and the row re-executes exactly once.
    executor_b = te.TaskExecutor(
        store=store,
        spec=spec,
        rows=[row_a, row_b, row_c],
        command_builder=lambda row: _ok_cmd(marker, row.row_id),
        run_id=run_id,
        resume=True,
        heartbeat_interval_s=1.0,
    )
    payload_b = executor_b.execute()
    assert payload_b["rows"][row_a.row_id]["status"] == "succeeded"
    assert payload_b["rows"][row_b.row_id]["status"] == "failed"
    assert payload_b["rows"][row_c.row_id]["status"] == "succeeded"

    lines = marker.read_text("utf-8").splitlines()
    assert lines.count("7:gp:research:0") == 1  # not re-executed
    assert "8:tpe:research:1" not in lines  # failed row never re-runs
    assert lines.count("9:rl:research:2") == 1  # unfinished row filled once

    ledger = _read_ledger(store, spec, run_id)
    attempts = _attempts_by_row(ledger)
    assert [a["status"] for a in attempts["7:gp:research:0"]] == ["succeeded"]
    assert [a["status"] for a in attempts["8:tpe:research:1"]] == ["failed"]
    assert [a["status"] for a in attempts["9:rl:research:2"]] == [
        "failed",  # interrupted attempt closed, evidence preserved
        "succeeded",  # re-execution after restart
    ]
    # The interrupted attempt's failure names the row and the reason.
    interrupted = next(
        entry
        for entry in ledger.iter_entries()
        if entry.status == "failed" and entry.error
        if "interrupted" in entry.error
    )
    assert "9:rl:research:2" in interrupted.error
    # Every row_id has at most one terminal success — no double execution.
    for att in attempts.values():
        assert [a["status"] for a in att].count("succeeded") <= 1


def test_resume_adopts_worker_result_of_interrupted_attempt_without_rerun(tmp_path):
    marker = tmp_path / "invocations.log"
    row_x = _rows((7, "gp", "0"))[0]
    store = RunStore(tmp_path / "store")
    spec = _spec()

    # Register the plan and complete the row once (terminal success).
    executor = te.TaskExecutor(
        store=store,
        spec=spec,
        rows=[row_x],
        command_builder=lambda row: _ok_cmd(marker, row.row_id),
        run_id="run-adopt",
    )
    executor.execute()
    assert marker.read_text("utf-8").splitlines() == ["7:gp:research:0"]

    # Simulate: a later session opened a new attempt, the worker left a
    # valid result file, but the supervisor died before closing the
    # trial.  Resume must adopt that result — never re-run the row.
    open_trial_id = _open_interrupted_attempt(store, spec, "run-adopt", row_x)
    te.write_row_result(
        executor.rows_dir / f"{te.row_slug(row_x)}.result.json",
        _worker_result_payload(row_x, "succeeded"),
    )
    executor_b = te.TaskExecutor(
        store=store,
        spec=spec,
        rows=[row_x],
        command_builder=lambda row: _ok_cmd(marker, row.row_id),
        run_id="run-adopt",
        resume=True,
        heartbeat_interval_s=1.0,
    )
    payload = executor_b.execute()
    assert payload["rows"][row_x.row_id]["status"] == "succeeded"
    # The row work itself never re-ran: still exactly one invocation.
    assert marker.read_text("utf-8").splitlines() == ["7:gp:research:0"]

    ledger = _read_ledger(store, spec, "run-adopt")
    attempts = _attempts_by_row(ledger)
    assert [a["status"] for a in attempts["7:gp:research:0"]].count("succeeded") == 2
    # The adopted attempt is closed with the worker's recorded metrics.
    adopted = next(
        entry
        for entry in ledger.iter_entries()
        if entry.trial_id == open_trial_id and entry.status != "running"
    )
    assert adopted.status == "succeeded"
    assert adopted.metrics["exit_code"] == 0
    assert adopted.metrics["peak_rss_mb"] == 20.0


# ------------------------------------------------------------ run identity


def test_run_identity_binding_single_writer_and_rerun_needs_new_run(tmp_path):
    store = RunStore(tmp_path / "store")
    spec = _spec()
    rows = _rows((7, "gp", "0"))

    executor = te.TaskExecutor(
        store=store,
        spec=spec,
        rows=rows,
        command_builder=lambda row: _ok_cmd(tmp_path / "a.log", row.row_id),
        run_id="run-identity",
    )
    # Single writer: a second executor on the SAME run is refused while
    # the first one holds the run's writer lock.
    with pytest.raises(RunLockedError):
        te.TaskExecutor(
            store=store,
            spec=spec,
            rows=rows,
            command_builder=lambda row: _ok_cmd(tmp_path / "b.log", row.row_id),
            run_id="run-identity",
            resume=True,
        )

    payload = executor.execute()
    run_dir = _run_dir(store, spec, "run-identity")
    for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        assert entry["spec_id"] == spec.spec_id
        assert entry["run_id"] == "run-identity"
    # The plan is registered as a content-addressed artifact of the run.
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    assert any(
        record["artifact_type"] == "task_executor_plan"
        for record in index["artifacts"]
    )

    # Rerun = new run id and a fresh output dir; the old run is untouched.
    before = (run_dir / "trials.jsonl").read_text(encoding="utf-8")
    executor_new = te.TaskExecutor(
        store=store,
        spec=spec,
        rows=rows,
        command_builder=lambda row: _ok_cmd(tmp_path / "c.log", row.row_id),
    )
    payload_new = executor_new.execute()
    assert payload_new["run_id"] != "run-identity"
    assert executor_new.rows_dir != run_dir / "executor_rows"
    assert (run_dir / "trials.jsonl").read_text(encoding="utf-8") == before

    # Plan drift on resume is refused fail-closed.
    with pytest.raises(te.TaskExecutorError):
        te.TaskExecutor(
            store=store,
            spec=spec,
            rows=_rows((7, "gp", "0"), (8, "tpe", "1")),
            command_builder=lambda row: _ok_cmd(tmp_path / "d.log", row.row_id),
            run_id="run-identity",
            resume=True,
        ).execute()
