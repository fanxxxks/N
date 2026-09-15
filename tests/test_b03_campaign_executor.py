"""P32 B03: resumable campaign execution with budget control.

Executable projection of ``docs/p32_b03_resumable_execution_budget_draft.md``
(authority chain: user direct instruction 2026-09-15 「完成B03」).  The
campaign adapter wires the R1 config legs onto the existing TaskExecutor
semantics (E3): every leg attempt is a ledger-bound TaskExecutor run with
an independent worker subprocess, stage telemetry, row budgets,
ownership-verified kills and explicit resume semantics, while the legacy
``.done`` markers degrade to derived display only.

Failure-test mapping (contract §4, R1-R9 = the plan's nine scenarios):
loading-phase crash, half-written state, duplicate start, killed
process tree, lost worker (heartbeat timeout), stale marker after an
identity change, partial config failure, budget exhaustion (row wall /
row RSS / campaign wall cap), user abort.  R10-R12 (rev 1, dual-review
dispositions): typed exit-2 refusals (identity resolution, config
preflight) and the --retry-failed drill; G7 pins the venv-shim subtree
rationale with a real grandchild interpreter.

Stub row commands are small real script files (worker/supervisor/state
all real); only the trivial exit/sleep helpers are one-line ``-c``
programs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import psutil
import pytest
import yaml

from ashare_model import campaign_executor as ce
from ashare_model import task_executor as te
from ashare_model.ledger import ExperimentLedger
from ashare_model.run_store import RunStore

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

_LEG_YAML = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "index_codes": ["000300.SH"],
    "min_listed_sessions": 1,
    "model": {"searcher": "random", "max_formula_len": 4},
    "backtest": {
        "train_end_date": "2024-02-06",
        "end_date": "2024-12-31",
        "top_n": 2,
        "initial_capital": 100000.0,
        "rebalance_frequency": "daily",
        "target_horizon": 1,
    },
    "protocol": {
        "folds": [{"train_end": "2024-02-06", "test_end": "2024-02-20"}],
        "seeds": [42],
        "random_samples": 0,
    },
}

_OK_STUB = """\
import pathlib, sys
marker, label = sys.argv[1], sys.argv[2]
pathlib.Path(marker).parent.mkdir(parents=True, exist_ok=True)
with open(marker, "a", encoding="utf-8") as fh:
    fh.write(label + "\\n")
"""

_PHASE_FAIL_STUB = """\
import json, os, pathlib, sys
marker, label, phase = sys.argv[1], sys.argv[2], sys.argv[3]
phase_path = os.environ["TASK_EXECUTOR_ROW_PHASE_FILE"]
with open(phase_path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"phase": phase, "entered_at": "t"}) + "\\n")
with open(marker, "a", encoding="utf-8") as fh:
    fh.write(label + "\\n")
raise SystemExit(3)
"""

_PHASE_OK_STUB = """\
import json, os, pathlib, sys
marker, label = sys.argv[1], sys.argv[2]
pathlib.Path(marker).parent.mkdir(parents=True, exist_ok=True)
with open(marker, "a", encoding="utf-8") as fh:
    fh.write(label + "\\n")
phase_path = os.environ.get("TASK_EXECUTOR_ROW_PHASE_FILE")
if phase_path:
    with open(phase_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"phase": "loading", "entered_at": "t"}) + "\\n")
"""

_ALLOC_STUB = """\
import sys, time
mb = int(sys.argv[1])
b = bytearray(mb * 1024 * 1024)
b[::4096] = b"\\x01" * len(range(0, len(b), 4096))
time.sleep(30)
"""

_WAIT_STUB = """\
import pathlib, sys, time
marker, signal_file, go, label = sys.argv[1:5]
pathlib.Path(marker).parent.mkdir(parents=True, exist_ok=True)
with open(marker, "a", encoding="utf-8") as fh:
    fh.write(label + "\\n")
pathlib.Path(signal_file).write_text("started", encoding="utf-8")
deadline = time.time() + 90
while not pathlib.Path(go).exists() and time.time() < deadline:
    time.sleep(0.05)
"""

_PHASES_STUB = """\
import json, os, pathlib, sys
marker = sys.argv[1]
pathlib.Path(marker).parent.mkdir(parents=True, exist_ok=True)
with open(marker, "a", encoding="utf-8") as fh:
    fh.write("row" + "\\n")
phase_path = os.environ["TASK_EXECUTOR_ROW_PHASE_FILE"]
with open(phase_path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"phase": "loading", "entered_at": "t1"}) + "\\n")
    fh.write(json.dumps({"phase": "search", "entered_at": "t2"}) + "\\n")
"""


def write_leg_config(tmp_path: Path, leg_id: str, **overrides) -> Path:
    """A real parseable leg YAML (no DB access needed to build the spec).

    All legs share one synthetic data block: a campaign shares one data
    identity (p32 §3.2/I4).  The A01 pre-flight verifies input existence;
    the stub leg commands never open the databases, so empty files
    satisfy the read-only resolver honestly.
    """

    cfg = json.loads(json.dumps(_LEG_YAML))
    for key, value in overrides.items():
        section, _, field_key = key.partition("__")
        if field_key:
            cfg.setdefault(section, {})[field_key] = value
        else:
            cfg[section] = value
    data_dir = tmp_path / "shared_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "ashare.duckdb").touch()
    (data_dir / "ashare_v2.duckdb").touch()
    cfg["data_dir"] = str(data_dir)
    cfg["duckdb_path"] = str(data_dir / "ashare.duckdb")
    cfg["parquet_dir"] = str(data_dir / "parquet")
    path = tmp_path / "configs" / f"{leg_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _stub(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / "stubs" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _cmd(script: Path, *args: object) -> list[str]:
    return [sys.executable, str(script), *[str(a) for a in args]]


def _identity() -> tuple[str, str]:
    return ("ds-b03-campaign-0001", "20240220")


def _manifest(
    tmp_path: Path,
    builders: dict,
    *,
    budgets: dict | None = None,
    name: str = "b03-test",
) -> ce.CampaignManifest:
    legs = []
    for leg_id, builder in builders.items():
        legs.append(
            ce.CampaignLeg(
                leg_id=leg_id,
                config=write_leg_config(tmp_path, leg_id),
                output=tmp_path / "r1" / leg_id / "protocol_result.json",
                ledger=tmp_path / "r1" / "ledger.jsonl",
                command=builder,
            )
        )
    return ce.CampaignManifest(
        name=name,
        legs=tuple(legs),
        **(budgets or {}),
    )


def _runner(tmp_path: Path, manifest: ce.CampaignManifest, **kwargs) -> ce.CampaignRunner:
    defaults = dict(
        repo_root=tmp_path,
        identity_provider=_identity,
        heartbeat_interval_s=0.5,
        heartbeat_timeout_s=60.0,
        poll_interval_s=0.05,
    )
    defaults.update(kwargs)
    return ce.CampaignRunner(
        manifest, tmp_path / "data" / "r1" / "campaign", **defaults
    )


def _state(tmp_path: Path) -> dict:
    return json.loads(
        (tmp_path / "data" / "r1" / "campaign" / "campaign_state.json").read_text(
            encoding="utf-8"
        )
    )


def _leg_run_dir(tmp_path: Path, state: dict, leg_id: str) -> Path:
    store = RunStore(tmp_path / "shared_data")
    record = state["legs"][leg_id]
    return store.runs_dir / record["spec_id"] / record["run_id"]


def _read_ledger(run_dir: Path) -> ExperimentLedger:
    spec_id = json.loads(
        (run_dir / "runspec.json").read_text(encoding="utf-8")
    ).get("spec_id")
    return ExperimentLedger(
        run_dir / "trials.jsonl",
        run_id=run_dir.name,
        spec_id=spec_id,
    )


def _fail_cmd(code: int = 3) -> list[str]:
    return [sys.executable, "-c", f"raise SystemExit({code})"]


def _sleep_cmd(seconds: float) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


# --------------------------------------------------------------------------
# R1: loading-phase crash — leg failure, active phase recorded, others fine
# --------------------------------------------------------------------------


def test_r1_loading_phase_crash_is_recorded_leg_failed_and_others_complete(tmp_path):
    marker = tmp_path / "markers.log"
    fail_stub = _stub(tmp_path, "phase_fail", _PHASE_FAIL_STUB)
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_a": lambda row: _cmd(fail_stub, marker, "leg_a", "loading"),
            "leg_b": lambda row: _cmd(ok_stub, marker, "leg_b"),
        },
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 1
    assert summary["campaign_status"] == "failed"
    assert summary["legs"]["leg_a"]["status"] == "failed"
    assert summary["legs"]["leg_b"]["status"] == "succeeded"
    assert summary["counts"] == {
        "total": 2,
        "succeeded": 1,
        "failed": 1,
        "aborted": 0,
        "reused": 0,
        "not_run": 0,
        "running": 0,
    }

    state = _state(tmp_path)
    # The active phase at death is evidence, not a guess.
    assert state["legs"]["leg_a"]["active_phase"] == "loading"
    run_dir = _leg_run_dir(tmp_path, state, "leg_a")
    ledger = _read_ledger(run_dir)
    failed = [e for e in ledger.iter_entries() if e.status == "failed"]
    assert failed and "leg_a" in (failed[-1].error or "")
    assert failed[-1].metrics.get("active_phase") == "loading"
    # The success was NOT staged as a whole-campaign pass.
    assert state["summary"]["exit_code"] == 1


# --------------------------------------------------------------------------
# R2: half-written campaign state — atomic authority keeps prior state
# --------------------------------------------------------------------------


def test_r2_injected_state_write_failure_leaves_prior_state_intact(
    tmp_path, monkeypatch
):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    runner = _runner(tmp_path, manifest)
    runner.run()
    before = _state(tmp_path)
    assert before["legs"]["leg_a"]["status"] == "succeeded"

    real_write = ce.atomic_write_json
    calls = {"n": 0}

    def flaky(path, payload):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("injected state write failure")
        return real_write(path, payload)

    monkeypatch.setattr(ce, "atomic_write_json", flaky)
    with pytest.raises(OSError):
        _runner(tmp_path, manifest).run()
    # The on-disk state is the last complete write, never a half file.
    after = _state(tmp_path)
    assert after["legs"]["leg_a"]["status"] == "succeeded"


# --------------------------------------------------------------------------
# R3: duplicate start — single writer, stale lock is an explicit action
# --------------------------------------------------------------------------


def test_r3_second_driver_is_refused_and_stale_lock_needs_explicit_clear(
    tmp_path,
):
    marker = tmp_path / "markers.log"
    go = marker.with_suffix(".go")
    started = marker.with_suffix(".started")
    wait_stub = _stub(tmp_path, "wait", _WAIT_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_a": lambda row: _cmd(
                wait_stub, marker, started, go, "leg_a"
            ),
        },
    )
    runner = _runner(tmp_path, manifest)
    result: dict = {}

    def drive():
        try:
            result["summary"] = runner.run()
        except BaseException as exc:  # surfaced to the main thread
            result["error"] = exc

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not started.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert started.exists(), "blocked leg never announced liveness"

    # A second driver on the same campaign is refused while the first holds
    # the campaign lock.
    with pytest.raises(ce.CampaignLockedError):
        _runner(tmp_path, manifest).run()

    # Release the leg and let the first driver finish.
    go.write_text("go", encoding="utf-8")
    thread.join(timeout=90)
    assert "error" not in result, result.get("error")
    assert result["summary"]["exit_code"] == 0
    assert not (
        tmp_path / "data" / "r1" / "campaign" / "campaign.lock"
    ).exists()

    # A stale lock (holder pid is dead) is refused by default...
    dead = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])
    dead.wait()
    # Windows pid reuse guard (review A#9): the drill's stale-holder pid
    # must actually be dead, not freshly recycled by an xdist neighbor.
    deadline = time.time() + 10
    while psutil.pid_exists(dead.pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(dead.pid)
    lock_path = tmp_path / "data" / "r1" / "campaign" / "campaign.lock"
    lock_path.write_text(
        json.dumps({"pid": dead.pid, "started_at": "t"}), encoding="utf-8"
    )
    with pytest.raises(ce.CampaignLockedError) as excinfo:
        _runner(tmp_path, manifest).run()
    assert "stale" in str(excinfo.value).lower()
    # ...and clearable only via the explicit, liveness-verified action.
    summary = _runner(tmp_path, manifest, clear_stale_lock=True).run()
    assert summary["exit_code"] == 0


# --------------------------------------------------------------------------
# R4: killed process tree — resume attaches the same run, evidence preserved
# --------------------------------------------------------------------------


def test_r4_killed_driver_tree_resumes_with_old_evidence_preserved(tmp_path):
    marker = tmp_path / "markers.log"
    go = marker.with_suffix(".go")
    started = marker.with_suffix(".started")
    write_leg_config(tmp_path, "leg_a")
    write_leg_config(tmp_path, "leg_b")
    write_leg_config(tmp_path, "leg_c")
    wait_stub = _stub(tmp_path, "wait", _WAIT_STUB)
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)

    phase_a = f"""
import sys
from pathlib import Path
from ashare_model import campaign_executor as ce
tmp = Path(r"{tmp_path}")
legs = [
    ce.CampaignLeg(leg_id='leg_a',
        config=Path(r'{tmp_path / "configs" / "leg_a.yaml"}'),
        output=Path(r'{tmp_path / "r1" / "leg_a" / "protocol_result.json"}'),
        ledger=Path(r'{tmp_path / "r1" / "ledger.jsonl"}'),
        command=[sys.executable, r'{ok_stub}', r'{marker}', 'leg_a']),
    ce.CampaignLeg(leg_id='leg_b',
        config=Path(r'{tmp_path / "configs" / "leg_b.yaml"}'),
        output=Path(r'{tmp_path / "r1" / "leg_b" / "protocol_result.json"}'),
        ledger=Path(r'{tmp_path / "r1" / "ledger.jsonl"}'),
        command=[sys.executable, r'{wait_stub}', r'{marker}', r'{started}', r'{go}', 'leg_b']),
]
manifest = ce.CampaignManifest(name="b03-r4", legs=tuple(legs))
runner = ce.CampaignRunner(
    manifest, tmp / "data" / "r1" / "campaign",
    repo_root=tmp, identity_provider=lambda: ("ds-b03-campaign-0001", "20240220"),
    heartbeat_interval_s=0.5, heartbeat_timeout_s=60.0, poll_interval_s=0.05,
)
runner.run()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", phase_a],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 90
    while not started.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert started.exists(), "phase A never reached the blocked leg"
    te._kill_tree(proc.pid, grace_s=5.0)
    proc.wait(timeout=30)

    state = _state(tmp_path)
    assert state["legs"]["leg_a"]["status"] == "succeeded"
    assert state["legs"]["leg_b"]["status"] == "running"
    old_run_dir = _leg_run_dir(tmp_path, state, "leg_b")
    ledger_before = (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")
    assert '"status":"running"' in ledger_before  # canonical compact JSON

    # The stale campaign lock is explicit: the holder pid is dead, verified.
    # The relaunch manifest keeps the SAME campaign name (review A#7: a
    # state file belongs to exactly one campaign).
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_a": lambda row: _cmd(ok_stub, marker, "leg_a"),
            "leg_b": lambda row: _cmd(wait_stub, marker, started, go, "leg_b"),
            "leg_c": lambda row: _cmd(ok_stub, marker, "leg_c"),
        },
        name="b03-r4",
    )
    summary = _runner(tmp_path, manifest, clear_stale_lock=True).run()
    assert summary["exit_code"] == 0
    assert summary["legs"]["leg_a"]["status"] == "succeeded"
    assert summary["legs"]["leg_a"]["reused"] is True
    assert summary["legs"]["leg_b"]["status"] == "succeeded"
    assert summary["legs"]["leg_c"]["status"] == "succeeded"

    # leg_b's run was ATTACHED (same run id) — the interrupted attempt is
    # closed as failed and the re-execution appended; evidence intact.
    state_after = _state(tmp_path)
    assert _leg_run_dir(tmp_path, state_after, "leg_b") == old_run_dir
    ledger_after = (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")
    assert ledger_after.startswith(ledger_before)
    ledger = _read_ledger(old_run_dir)
    statuses = [e.status for e in ledger.iter_entries()]
    assert statuses == ["running", "failed", "running", "succeeded"]


# --------------------------------------------------------------------------
# R5: lost worker — heartbeat timeout fails the leg, campaign continues
# --------------------------------------------------------------------------


def test_r5_lost_worker_heartbeat_timeout_fails_leg_and_continues(tmp_path):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_slow": lambda row: _sleep_cmd(120),
            "leg_ok": lambda row: _cmd(ok_stub, marker, "leg_ok"),
        },
    )
    summary = _runner(
        tmp_path,
        manifest,
        # Lost contact = the worker's beats stop: the interval exceeds the
        # timeout so only the initial beat lands (the sleeping row command
        # itself keeps the worker alive — beats must NOT depend on it).
        heartbeat_interval_s=60.0,
        heartbeat_timeout_s=15.0,
    ).run()
    assert summary["exit_code"] == 1
    assert summary["legs"]["leg_slow"]["status"] == "failed"
    assert "heartbeat timeout" in summary["legs"]["leg_slow"]["error"]
    assert summary["legs"]["leg_ok"]["status"] == "succeeded"


# --------------------------------------------------------------------------
# R6: stale marker / identity change — .done is never the authority
# --------------------------------------------------------------------------


def test_r6a_config_change_invalidates_reuse_and_opens_a_new_run(tmp_path):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    config = write_leg_config(tmp_path, "leg_x")
    leg = ce.CampaignLeg(
        leg_id="leg_x",
        config=config,
        output=tmp_path / "r1" / "leg_x" / "protocol_result.json",
        ledger=tmp_path / "r1" / "ledger.jsonl",
        command=lambda row: _cmd(ok_stub, marker, "leg_x"),
    )
    manifest = ce.CampaignManifest(name="b03-r6", legs=(leg,))
    assert _runner(tmp_path, manifest).run()["exit_code"] == 0
    state = _state(tmp_path)
    old_run_dir = _leg_run_dir(tmp_path, state, "leg_x")
    ledger_before = (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")
    assert marker.read_text("utf-8").splitlines() == ["leg_x"]

    # The config content changes: the succeeded identity is stale.
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["model"]["max_formula_len"] = 5
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 0
    assert summary["legs"]["leg_x"]["reused"] is False
    assert marker.read_text("utf-8").splitlines() == ["leg_x", "leg_x"]

    state_after = _state(tmp_path)
    assert (
        state_after["legs"]["leg_x"]["identity_hash"]
        != state["legs"]["leg_x"]["identity_hash"]
    )
    new_run_dir = _leg_run_dir(tmp_path, state_after, "leg_x")
    assert new_run_dir != old_run_dir
    # The old run's ledger was never touched (old evidence preserved).
    assert (
        (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")
        == ledger_before
    )


def test_r6b_bare_done_marker_without_state_is_not_reuse_authority(tmp_path):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    done = tmp_path / "data" / "r1" / "leg_a.done"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text("legacy marker with no identity\n", encoding="utf-8")
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 0
    assert summary["legs"]["leg_a"]["reused"] is False
    # The leg ran despite the marker: the marker is display only.
    assert marker.read_text("utf-8").splitlines() == ["leg_a"]


# --------------------------------------------------------------------------
# R7: partial config failure — never a whole-campaign success
# --------------------------------------------------------------------------


def test_r7_partial_failure_keeps_exit_1_and_runs_remaining_legs(tmp_path):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_1": lambda row: _cmd(ok_stub, marker, "leg_1"),
            "leg_2": lambda row: _fail_cmd(),
            "leg_3": lambda row: _cmd(ok_stub, marker, "leg_3"),
        },
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 1
    assert summary["campaign_status"] == "failed"
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["succeeded"] == 2
    assert sorted(marker.read_text("utf-8").splitlines()) == ["leg_1", "leg_3"]


# --------------------------------------------------------------------------
# R8: budget exhaustion — row wall / row RSS / campaign wall cap
# --------------------------------------------------------------------------


def test_r8a_row_wall_clock_budget_kills_leg_with_budget_reason(tmp_path):
    manifest = _manifest(
        tmp_path,
        {"leg_slow": lambda row: _sleep_cmd(30)},
        budgets={"row_wall_clock_budget_s": 1.5},
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 1
    assert summary["legs"]["leg_slow"]["status"] == "failed"
    error = summary["legs"]["leg_slow"]["error"]
    assert "wall-clock budget" in error
    assert "heartbeat timeout" not in error  # distinct meaning


def test_r8b_row_peak_rss_budget_kills_leg_with_memory_reason(tmp_path):
    alloc_stub = _stub(tmp_path, "alloc", _ALLOC_STUB)
    manifest = _manifest(
        tmp_path,
        {"leg_fat": lambda row: _cmd(alloc_stub, 200)},
        budgets={"row_peak_rss_budget_mb": 60.0},
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 1
    assert summary["legs"]["leg_fat"]["status"] == "failed"
    assert "peak RSS budget" in summary["legs"]["leg_fat"]["error"]


def test_r8c_campaign_wall_cap_marks_remaining_legs_not_run(tmp_path):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_1": lambda row: _cmd(ok_stub, marker, "leg_1"),
            "leg_2": lambda row: _cmd(ok_stub, marker, "leg_2"),
        },
        budgets={"campaign_wall_clock_budget_s": 1e-6},
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 1
    assert summary["campaign_status"] == "stopped_budget"
    assert summary["legs"]["leg_1"]["status"] == "not_run"
    assert "budget" in summary["legs"]["leg_1"]["reason"]
    assert marker.exists() is False


# --------------------------------------------------------------------------
# R9: user abort — distinct meaning, worker tree dead, resume re-executes
# --------------------------------------------------------------------------


def test_r9_user_abort_records_aborted_kills_worker_and_resume_reexecutes(
    tmp_path,
):
    marker = tmp_path / "markers.log"
    started = marker.with_suffix(".started")
    go = marker.with_suffix(".go")  # never written: the stub blocks until killed
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    wait_stub = _stub(tmp_path, "wait", _WAIT_STUB)
    manifest = _manifest(
        tmp_path,
        {
            "leg_a": lambda row: _cmd(wait_stub, marker, started, go, "leg_a"),
            "leg_b": lambda row: _cmd(ok_stub, marker, "leg_b"),
        },
    )
    runner = _runner(tmp_path, manifest)
    # Deterministic abort: interrupt only AFTER the in-flight leg
    # announced liveness (a pre-leg interrupt aborts nothing).
    timer = threading.Timer(0.0, _interrupt_when, args=(started,))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run()
    finally:
        timer.cancel()

    state = _state(tmp_path)
    assert state["legs"]["leg_a"]["status"] == "aborted"
    assert state["summary"]["exit_code"] == 130
    run_dir = _leg_run_dir(tmp_path, state, "leg_a")
    # Review A#1/A#4: the abort must actually KILL the worker tree — read
    # the worker's self-announced pids from its heartbeat and assert both
    # the worker and its row command are gone (contract §4/R9: the worker
    # tree is indeed dead).
    import psutil

    heartbeat = json.loads(
        next(iter((run_dir / "executor_rows").glob("*.heartbeat.json"))).read_text(
            "utf-8"
        )
    )
    deadline = time.time() + 30
    while (
        psutil.pid_exists(heartbeat["pid"]) or psutil.pid_exists(heartbeat["child_pid"])
    ) and time.time() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(heartbeat["pid"]), "worker survived the abort"
    assert not psutil.pid_exists(heartbeat["child_pid"]), (
        "row command survived the abort"
    )
    ledger = _read_ledger(run_dir)
    # The interrupted attempt stays open in the ledger (append-only
    # evidence; the resume closes it as failed and re-executes).
    assert [e.status for e in ledger.iter_entries()] == ["running"]

    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 0
    assert summary["legs"]["leg_a"]["status"] == "succeeded"
    assert summary["legs"]["leg_b"]["status"] == "succeeded"
    ledger_after = _read_ledger(run_dir)
    assert [e.status for e in ledger_after.iter_entries()] == [
        "running",
        "failed",
        "running",
        "succeeded",
    ]


def _interrupt_main() -> None:
    import _thread

    _thread.interrupt_main()


def _interrupt_when(signal_file: Path, timeout: float = 60.0) -> None:
    """Interrupt the main thread only after the in-flight leg announced
    liveness — makes the user-abort drill deterministic."""

    deadline = time.time() + timeout
    while not signal_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    _interrupt_main()


# --------------------------------------------------------------------------
# Guards: reuse, telemetry, stage plan, executor phase plumbing, manifest
# --------------------------------------------------------------------------


def test_g1_identity_match_reuses_succeeded_leg_and_writes_derived_done(
    tmp_path,
):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    assert _runner(tmp_path, manifest).run()["exit_code"] == 0
    assert marker.read_text("utf-8").splitlines() == ["leg_a"]

    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 0
    assert summary["legs"]["leg_a"]["reused"] is True
    assert summary["counts"]["reused"] == 1
    # The command did NOT run again.
    assert marker.read_text("utf-8").splitlines() == ["leg_a"]
    # The derived marker exists and is written from terminal success.
    done = tmp_path / "data" / "r1" / "leg_a.done"
    assert done.exists()
    assert "leg_a" in done.read_text(encoding="utf-8")


def test_g2_telemetry_extracted_from_leg_protocol_artifact(tmp_path):
    output = tmp_path / "r1" / "leg_a" / "protocol_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "n_candidates": 7,
                "rows": [
                    {
                        "search_result": {
                            "requested_budget": 10,
                            "consumed_budget": 8,
                            "proposal_count": 12,
                            "invalid_proposals": 2,
                            "semantic_duplicates": 2,
                        }
                    },
                    {
                        "search_result": {
                            "requested_budget": 10,
                            "consumed_budget": 9,
                            "proposal_count": 11,
                            "invalid_proposals": 1,
                            "semantic_duplicates": 1,
                        }
                    },
                    {"no_search_result_here": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 0
    telemetry = _state(tmp_path)["legs"]["leg_a"]["telemetry"]
    assert telemetry["requested_budget"] == 20
    assert telemetry["consumed_budget"] == 17
    assert telemetry["proposal_count"] == 23
    assert telemetry["invalid_proposals"] == 3
    assert telemetry["semantic_duplicates"] == 3
    assert telemetry["n_candidates"] == 7


def test_g3_trial_payload_declares_the_five_stage_plan(tmp_path):
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    _runner(tmp_path, manifest).run()
    state = _state(tmp_path)
    ledger = _read_ledger(_leg_run_dir(tmp_path, state, "leg_a"))
    opening = next(e for e in ledger.iter_entries() if e.status == "running")
    assert opening.payload["leg_id"] == "leg_a"
    assert opening.payload["stages"] == [
        "loading",
        "feature_build",
        "search",
        "evaluation",
        "artifact_write",
    ]


def test_g4_executor_records_phase_file_into_result_and_ledger(tmp_path):
    """Executor-level phase plumbing (env var -> result -> ledger)."""

    marker = tmp_path / "invocations.log"
    phases_stub = _stub(tmp_path, "phases", _PHASES_STUB)
    from ashare_data.config import BacktestConfig
    from ashare_model.runspec import WindowSpec, build_runspec
    from ashare_portfolio.execution_spec import portfolio_config_provenance

    spec = build_runspec(
        dataset_id="ds-b03-exec-0001",
        data_cutoff="20241231",
        frequency="daily",
        target="forward_return",
        horizon=1,
        requested_budget=10,
        n_folds=1,
        dev_window=WindowSpec(start="20150101", end="20231231"),
        preregistered_contract_hash="a" * 64,
        resolved_thresholds={"min_eligible": 3.0},
        portfolio_config=portfolio_config_provenance(BacktestConfig()),
        require_clean_tree=False,
    )
    store = RunStore(tmp_path / "store")
    executor = te.TaskExecutor(
        store=store,
        spec=spec,
        rows=[te.TaskRow(seed=1, backend="evaluation", track="r1", fold="leg")],
        command_builder=lambda row: _cmd(phases_stub, marker),
        run_id="run-phases",
        heartbeat_interval_s=1.0,
    )
    executor.execute()
    result = json.loads(
        (executor.rows_dir / "1_evaluation_r1_leg.result.json").read_text("utf-8")
    )
    assert result["status"] == "succeeded"
    assert result["phases"] == [
        {"phase": "loading", "entered_at": "t1"},
        {"phase": "search", "entered_at": "t2"},
    ]
    assert result["phase"] == "search"
    ledger = ExperimentLedger(
        store.runs_dir / spec.spec_id / "run-phases" / "trials.jsonl",
        run_id="run-phases",
        spec_id=spec.spec_id,
    )
    terminal = next(e for e in ledger.iter_entries() if e.status != "running")
    assert terminal.metrics["active_phase"] == "search"
    assert len(terminal.metrics["phases"]) == 2


def test_g5_r1_manifest_builds_six_parseable_legs():
    """The R1 manifest (calibration + five configs) parses real YAML."""

    from ashare_data.config import (
        load_config,
        make_backtest_config,
        make_data_config,
        make_model_config,
        make_protocol_config,
    )
    from scripts.r1_campaign import build_r1_manifest

    repo_root = Path(__file__).resolve().parents[1]
    manifest = build_r1_manifest(repo_root)
    assert [leg.leg_id for leg in manifest.legs] == [
        "calibration",
        "r1_medium_a",
        "r1_medium_ab",
        "r1_slow_ab",
        "r1_short_a",
        "r1_unified_base",
    ]
    for leg in manifest.legs:
        assert leg.command is None  # standard evaluation command path
        raw = load_config(leg.config, project_root=repo_root)
        make_data_config(raw, repo_root)
        make_model_config(raw)
        make_backtest_config(raw)
        make_protocol_config(raw)


def test_g6_real_evaluation_records_all_five_stage_phases(
    tmp_path, populated_db, production_gate_asof, monkeypatch
):
    """The real evaluation entry writes the five stage milestones in order
    (contract §3.1) — verified on a real closed-loop run, never by source
    string inspection."""

    from ashare_model import evaluation as evaluation_module

    data_config = populated_db
    cfg = json.loads(json.dumps(_LEG_YAML))
    cfg["data_dir"] = str(data_config.data_dir)
    cfg["duckdb_path"] = str(data_config.duckdb_path)
    cfg["parquet_dir"] = str(data_config.parquet_dir)
    cfg["protocol"]["screening"] = {"steps": 1, "batch_size": 2}
    cfg["protocol"]["gp_enabled"] = False
    cfg["protocol"]["tpe_enabled"] = False
    config_path = tmp_path / "leg_eval.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    phase_file = tmp_path / "row.phases.jsonl"
    monkeypatch.setenv("TASK_EXECUTOR_ROW_PHASE_FILE", str(phase_file))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "--config", str(config_path),
            "--tier", "screening",
            "--output", str(tmp_path / "protocol_result.json"),
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--regime", str(tmp_path / "regime.json"),
            "--min-eligible", "3",
            "--lifecycle-mode", "engineering",
        ],
    )
    assert evaluation_module.main() == 0
    phases = [
        json.loads(line)["phase"]
        for line in phase_file.read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert phases == [
        "loading",
        "feature_build",
        "search",
        "evaluation",
        "artifact_write",
    ]


# --------------------------------------------------------------------------
# rev 1 dispositions (dual review A#2/A#3, B#1/B#2/B#6): typed refusals,
# reuse identity hardening, retry-failed drill, grandchild subtree kill
# --------------------------------------------------------------------------


def test_r10_identity_resolution_refusal_is_exit_2(tmp_path):
    """DatasetIdMismatch (and any identity-resolution failure) is an
    OPERATIONAL refusal: CampaignExecutorError (exit 2), never a leg
    failure and never a raw traceback (p32 §3.4 rev 1)."""

    from ashare_data.manifest import DatasetIdMismatch

    def refusing_identity():
        raise DatasetIdMismatch("no complete composite generation exists")

    marker = tmp_path / "m.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    with pytest.raises(ce.CampaignExecutorError) as excinfo:
        _runner(
            tmp_path, manifest, identity_provider=refusing_identity
        ).run()
    assert "identity resolution refused" in str(excinfo.value)
    # Nothing ran: no state was created, no leg executed.
    assert not (
        tmp_path / "data" / "r1" / "campaign" / "campaign_state.json"
    ).exists()
    assert not marker.exists()


def test_r11_config_preflight_refusal_is_exit_2(tmp_path):
    """The A01 pre-flight refusal is an OPERATIONAL refusal (p32 §3.4
    rev 1, review B#2): it stops the campaign with exit 2 semantics
    instead of being recorded as a leg failure."""

    marker = tmp_path / "m.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    # Remove the required input the write_leg_config fixture touched.
    (tmp_path / "shared_data" / "ashare_v2.duckdb").unlink()
    with pytest.raises(ce.CampaignExecutorError) as excinfo:
        _runner(tmp_path, manifest).run()
    assert "pre-flight" in str(excinfo.value)
    # The leg never started.
    state = _state(tmp_path)
    assert not state["legs"].get("leg_a", {}).get("run_id")
    assert not marker.exists()


def test_r12_retry_failed_opens_new_run_and_preserves_old_evidence(tmp_path):
    """p32 §3.5/I7 drill (review B#6): a terminal-failed leg is skipped
    with its reason by default; --retry-failed opens a NEW run while the
    old run's ledger stays byte-identical."""

    marker = tmp_path / "markers.log"
    fail_stub = _stub(tmp_path, "fail3", "raise SystemExit(3)\n")
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(fail_stub, marker, "leg_a")}
    )
    assert _runner(tmp_path, manifest).run()["exit_code"] == 1
    state = _state(tmp_path)
    old_run_dir = _leg_run_dir(tmp_path, state, "leg_a")
    ledger_before = (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")

    # Default relaunch: skipped with the preservation reason, no new run.
    summary = _runner(tmp_path, manifest).run()
    assert summary["legs"]["leg_a"]["status"] == "failed"
    assert "retry-failed" in summary["legs"]["leg_a"]["reason"]
    assert _leg_run_dir(tmp_path, _state(tmp_path), "leg_a") == old_run_dir
    assert (
        (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")
        == ledger_before
    )

    # The explicit retry: NEW run id, old ledger untouched, leg succeeds.
    manifest_retry = _manifest(
        tmp_path,
        {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")},
    )
    summary = _runner(tmp_path, manifest_retry, retry_failed=True).run()
    assert summary["exit_code"] == 0
    assert summary["legs"]["leg_a"]["status"] == "succeeded"
    state_after = _state(tmp_path)
    assert (
        state_after["legs"]["leg_a"]["run_id"]
        != state["legs"]["leg_a"]["run_id"]
    )
    assert _leg_run_dir(tmp_path, state_after, "leg_a") != old_run_dir
    assert (
        (old_run_dir / "trials.jsonl").read_text(encoding="utf-8")
        == ledger_before
    )


def test_g7_subtree_rss_and_tree_kill_cover_a_real_grandchild(tmp_path):
    """The venv-shim rationale (measurement log §3.1) as a test: the row
    command spawns a REAL grandchild interpreter; the budget kill takes
    the whole tree down including the grandchild (review A#13)."""

    marker = tmp_path / "markers.log"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    stop = tmp_path / "stop"
    spawn_stub = _stub(
        tmp_path,
        "spawner",
        """\
import pathlib, subprocess, sys, time
grandchild = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import os, pathlib, time; "
        "pathlib.Path(r'" + sys.argv[2] + "').write_text(str(os.getpid())); "
        "time.sleep(120)",
    ],
)
pathlib.Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
with open(sys.argv[1], "a", encoding="utf-8") as fh:
    fh.write("spawner" + chr(10))
deadline = time.time() + 60
while not pathlib.Path(sys.argv[3]).exists() and time.time() < deadline:
    time.sleep(0.05)
""",
    )
    manifest = _manifest(
        tmp_path,
        {
            "leg_tree": lambda row: _cmd(
                spawn_stub, marker, grandchild_pid_file, stop
            ),
        },
        budgets={"row_wall_clock_budget_s": 2.0},
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 1
    assert summary["legs"]["leg_tree"]["status"] == "failed"
    assert "wall-clock budget" in summary["legs"]["leg_tree"]["error"]
    # The wall budget killed the tree: worker, row command AND the real
    # grandchild interpreter are all gone.
    state = _state(tmp_path)
    run_dir = _leg_run_dir(tmp_path, state, "leg_tree")
    heartbeat = json.loads(
        next(iter((run_dir / "executor_rows").glob("*.heartbeat.json"))).read_text(
            "utf-8"
        )
    )
    deadline = time.time() + 30
    while time.time() < deadline and (
        psutil.pid_exists(heartbeat["pid"])
        or psutil.pid_exists(heartbeat["child_pid"])
    ):
        time.sleep(0.1)
    assert not psutil.pid_exists(heartbeat["pid"])
    assert not psutil.pid_exists(heartbeat["child_pid"])
    grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8"))
    deadline = time.time() + 30
    while psutil.pid_exists(grandchild_pid) and time.time() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(grandchild_pid), (
        "the grandchild interpreter survived the budget tree kill"
    )


def test_g2b_telemetry_partial_missing_fields_stay_null(tmp_path):
    """I11 (review A#12): a field is null unless its source is present —
    a row without a field never fabricates a 0."""

    output = tmp_path / "r1" / "leg_a" / "protocol_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "n_candidates": 5,
                "rows": [
                    {"search_result": {"consumed_budget": 8}},
                    {"no_search_result_here": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "markers.log"
    ok_stub = _stub(tmp_path, "ok", _OK_STUB)
    manifest = _manifest(
        tmp_path, {"leg_a": lambda row: _cmd(ok_stub, marker, "leg_a")}
    )
    summary = _runner(tmp_path, manifest).run()
    assert summary["exit_code"] == 0
    telemetry = _state(tmp_path)["legs"]["leg_a"]["telemetry"]
    assert telemetry["consumed_budget"] == 8
    assert telemetry["requested_budget"] is None
    assert telemetry["proposal_count"] is None
    assert telemetry["invalid_proposals"] is None
    assert telemetry["semantic_duplicates"] is None
    assert telemetry["n_candidates"] == 5
