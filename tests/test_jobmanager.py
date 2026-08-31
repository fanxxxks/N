from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ashare_data.config import SimConfig
from ashare_trading.manager import (
    RunConflictError,
    SimJobManager,
    build_run_sim_argv,
)

TERMINAL = ("stopped", "finished", "error")


def _wait_for(predicate, timeout: float = 30.0) -> None:
    # 30s ceiling: subprocess spawn/kill cycles can exceed 10s when the
    # machine runs heavy jobs (protocol + sync) in parallel, and a
    # satisfied predicate returns immediately regardless.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("condition not met within timeout")


def _sleep_cmd(seconds: int = 300) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _progress_then_sleep_cmd(progress_path: Path, seconds: int = 300) -> list[str]:
    code = (
        "import json, pathlib, time; "
        f"pathlib.Path({str(progress_path)!r}).parent.mkdir(parents=True, exist_ok=True); "
        f"pathlib.Path({str(progress_path)!r}).write_text("
        "json.dumps({'phase': 'executing', 'current_date': '20240102', "
        "'equity': 123.0})); "
        f"time.sleep({seconds})"
    )
    return [sys.executable, "-c", code]


def _finish_cmd(progress_path: Path) -> list[str]:
    code = (
        "import json, pathlib; "
        f"pathlib.Path({str(progress_path)!r}).parent.mkdir(parents=True, exist_ok=True); "
        f"pathlib.Path({str(progress_path)!r}).write_text("
        "json.dumps({'phase': 'finished', 'current_date': '20240110', "
        "'equity': 456.0}))"
    )
    return [sys.executable, "-c", code]


def _manager(tmp_path: Path, cmd_builder=None, **kwargs) -> SimJobManager:
    sim_config = SimConfig(
        initial_capital=100000.0,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
        stop_signal_path=tmp_path / "STOP",
        progress_path=tmp_path / "progress.json",
    )
    # Short graces so stop-escalation tests run fast; keep the sleep long
    # enough that a finished test never leaves an orphan behind.
    kwargs.setdefault("stop_grace", 0.3)
    kwargs.setdefault("kill_grace", 0.3)
    return SimJobManager(
        root=tmp_path,
        sim_config=sim_config,
        cmd_builder=cmd_builder,
        run_file=tmp_path / "run.json",
        lock_file=tmp_path / "run.lock",
        **kwargs,
    )


def _stop_and_wait(manager: SimJobManager) -> None:
    manager.stop()
    _wait_for(lambda: manager.status()["state"] in TERMINAL)


# --------------------------------------------------------------- argv building


def test_build_run_sim_argv_reset_and_resume(tmp_path: Path):
    # No state: plain replay (the runner fails fast on existing history).
    argv = build_run_sim_argv(state_path=tmp_path / "nope.json")
    assert "--resume" not in argv and "--reset" not in argv

    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    argv = build_run_sim_argv(state_path=state)
    assert "--resume" in argv and "--reset" not in argv

    argv = build_run_sim_argv(reset=True, state_path=state)
    assert "--reset" in argv and "--resume" not in argv

    argv = build_run_sim_argv(
        start_date="2024-01-05", end_date="2024-02-01", state_path=state
    )
    assert argv[-4:] == ["--start", "2024-01-05", "--end", "2024-02-01"]


# ---------------------------------------------------------------------- start


def test_start_and_status_running(tmp_path: Path):
    progress = tmp_path / "progress.json"
    manager = _manager(
        tmp_path,
        cmd_builder=lambda _reset, _start, _end: _progress_then_sleep_cmd(progress),
    )
    status = manager.start()
    assert status["state"] in ("starting", "running")
    assert status["pid"]

    _wait_for(lambda: manager.status()["state"] == "running")
    status = manager.status()
    assert status["state"] == "running"
    assert status["current_date"] == "20240102"
    assert status["equity"] == 123.0
    assert status["log_path"] and Path(status["log_path"]).exists()
    _stop_and_wait(manager)


def test_start_conflict_while_running(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    manager.start()
    with pytest.raises(RunConflictError):
        manager.start()
    _stop_and_wait(manager)


def test_start_conflict_survives_second_boundary_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression (PR3 parity, 2026-08-31): sample started_at before Popen.

    The PID-reuse guard in ``_is_alive`` rejects processes whose create_time
    is older than the recorded ``started_at``. ``started_at`` used to be
    sampled after Popen returned, so any spawn whose Popen->sample window
    crossed a clock-second boundary — routine under parallel-worker load —
    made the guard reject the manager's own live child; a second ``start()``
    then stole the lock and double-spawned instead of raising
    RunConflictError. Deterministic reproduction: a post-Popen delay pushes
    the ``started_at`` sample into the next clock second unconditionally.
    """

    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    real_popen = subprocess.Popen

    def slow_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        time.sleep(1.1)  # force the started_at sample past the child's second
        return proc

    monkeypatch.setattr(subprocess, "Popen", slow_popen)
    manager.start()
    with pytest.raises(RunConflictError):
        manager.start()
    monkeypatch.undo()
    _stop_and_wait(manager)


def test_start_clears_leftover_stop_signal_and_progress(tmp_path: Path):
    stop = tmp_path / "STOP"
    stop.write_text("STOPPED", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps({"phase": "finished"}), encoding="utf-8"
    )
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    manager.start()
    assert not stop.exists()
    assert not (tmp_path / "progress.json").exists()
    _stop_and_wait(manager)


def test_start_steals_stale_lock_and_dead_record(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    (tmp_path / "run.lock").write_text("999999", encoding="utf-8")
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": 999999,
                "started_at": "2000-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    status = manager.start()
    assert status["pid"] and status["pid"] != 999999
    _stop_and_wait(manager)


def test_start_refuses_live_lock_holder(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    manager.start()
    holder_pid = manager.status()["pid"]
    # A second manager instance sees the live record and refuses.
    other = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    with pytest.raises(RunConflictError):
        other.start()
    assert manager.status()["pid"] == holder_pid
    _stop_and_wait(manager)


# ----------------------------------------------------------------------- stop


def test_stop_escalates_after_grace(tmp_path: Path):
    manager = _manager(
        tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd(600)
    )
    manager.start()
    result = manager.stop()
    assert result["ok"] and result["state"] == "stopping"
    _wait_for(lambda: manager.status()["state"] == "stopped")
    status = manager.status()
    assert status["state"] == "stopped"
    assert not (tmp_path / "run.lock").exists()


def test_stop_when_idle_is_a_noop(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    result = manager.stop()
    assert result["ok"]
    assert result["state"] == "idle"


# --------------------------------------------------------------------- status


def test_status_reconciles_finished_process(tmp_path: Path):
    progress = tmp_path / "progress.json"
    manager = _manager(
        tmp_path, cmd_builder=lambda _reset, _start, _end: _finish_cmd(progress)
    )
    manager.start()
    _wait_for(lambda: manager.status()["state"] == "finished")
    status = manager.status()
    assert status["state"] == "finished"
    assert status["current_date"] == "20240110"
    assert status["equity"] == 456.0
    assert not (tmp_path / "run.lock").exists()


def test_status_reconciles_crash_as_error(tmp_path: Path):
    cmd = [sys.executable, "-c", "raise SystemExit(3)"]
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: cmd)
    manager.start()
    _wait_for(lambda: manager.status()["state"] == "error")
    status = manager.status()
    assert status["state"] == "error"
    assert "without a terminal progress phase" in status["error"]


def test_status_idle_without_record(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    status = manager.status()
    assert status["state"] == "idle"
    assert status["pid"] is None


# ---------------------------------------------------------------------- reset


def _write_history_state(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "initial_capital": 100000.0,
                "cash": 50000.0,
                "trade_count": 3,
                "last_exec_date": "20240105",
                "positions": {},
                "equity_history": [{"trade_date": "20240105", "equity": 100000.0}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "orders").mkdir()
    (tmp_path / "orders" / "20240105.json").write_text("[]", encoding="utf-8")
    (tmp_path / "trades").mkdir()
    (tmp_path / "trades" / "20240105.json").write_text("[]", encoding="utf-8")


def test_reset_archives_then_resets(tmp_path: Path):
    marker = tmp_path / "archived.marker"
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "archive_run.py").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ok')",
        encoding="utf-8",
    )
    _write_history_state(tmp_path)
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    result = manager.reset()
    assert result["ok"], result
    assert marker.exists()

    # P8-05 reset removes the convenience state; immutable v2 evidence
    # remains content-addressed in RunStore and legacy bytes were archived.
    assert not (tmp_path / "state.json").exists()
    # The paper trail is parked, never deleted.
    assert not (tmp_path / "orders").exists()
    assert list(tmp_path.glob("orders.bak_*"))
    assert not (tmp_path / "trades").exists()
    assert list(tmp_path.glob("trades.bak_*"))


def test_reset_aborts_when_archive_fails(tmp_path: Path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "archive_run.py").write_text(
        "import sys; sys.exit(1)", encoding="utf-8"
    )
    _write_history_state(tmp_path)
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    result = manager.reset()
    assert not result["ok"]
    assert "archive failed" in result["reason"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["trade_count"] == 3  # untouched
    assert (tmp_path / "orders").exists()


def test_reset_skips_archive_without_history(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    result = manager.reset()
    assert result["ok"]
    assert result["archive"] == "no history to archive"
    assert not (tmp_path / "state.json").exists()


def test_reset_conflict_while_running(tmp_path: Path):
    manager = _manager(tmp_path, cmd_builder=lambda _reset, _start, _end: _sleep_cmd())
    manager.start()
    with pytest.raises(RunConflictError):
        manager.reset()
    _stop_and_wait(manager)
