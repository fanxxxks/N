"""Process-level manager for the daily paper-trading runner.

All state lives on disk (``data/sim_run.json`` + a lock file) so the manager
survives ``uvicorn --reload`` restarts and multiple API processes share one
source of truth. Statuses:

    idle -> starting -> running -> stopping -> stopped / finished / error

Stop escalation: ``STOP_SIGNAL`` first (the runner checks it between trading
days), then ``terminate``, then ``kill`` after grace periods. On Windows the
child is spawned in its own process group so a Ctrl+C on the API console can
never kill the simulation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import psutil
from loguru import logger

from ashare_data.config import SimConfig

TERMINAL_STATES = ("stopped", "finished", "error")
DEFAULT_STOP_GRACE_SECONDS = 30.0
DEFAULT_KILL_GRACE_SECONDS = 5.0


class RunConflictError(RuntimeError):
    """A simulation run is already active."""


def build_run_sim_argv(
    reset: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    state_path: str | Path | None = None,
) -> list[str]:
    """The fixed argv used to launch the runner (no shell, no free-form args)."""

    argv = [sys.executable, "-m", "ashare_trading.run_sim"]
    if reset:
        argv.append("--reset")
    elif state_path and Path(state_path).exists():
        # Replay safety: an existing state can only be continued, never
        # silently replayed (the runner itself also enforces this).
        argv.append("--resume")
    if start_date:
        argv += ["--start", start_date]
    if end_date:
        argv += ["--end", end_date]
    return argv


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class SimJobManager:
    def __init__(
        self,
        root: str | Path,
        sim_config: SimConfig | None = None,
        run_file: str | Path | None = None,
        lock_file: str | Path | None = None,
        cmd_builder: Callable[[], list[str]] | None = None,
        stop_grace: float = DEFAULT_STOP_GRACE_SECONDS,
        kill_grace: float = DEFAULT_KILL_GRACE_SECONDS,
    ) -> None:
        self.root = Path(root)
        self.sim_config = sim_config
        self.run_file = Path(run_file) if run_file else self.root / "data" / "sim_run.json"
        self.lock_file = Path(lock_file) if lock_file else self.root / "data" / "sim_run.lock"
        self.cmd_builder = cmd_builder or self._default_cmd
        self.stop_grace = stop_grace
        self.kill_grace = kill_grace
        self._escalation_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ state

    def _default_cmd(self) -> list[str]:
        state_path = self.sim_config.state_path if self.sim_config else None
        return build_run_sim_argv(state_path=state_path)

    def _read_run(self) -> dict | None:
        try:
            if not self.run_file.exists():
                return None
            payload = json.loads(self.run_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:  # noqa: BLE001 - half-written file must not kill the API
            return None

    def _write_run(self, payload: dict) -> None:
        _atomic_write_json(self.run_file, payload)

    def _progress(self) -> dict:
        if self.sim_config is None:
            return {}
        try:
            path = Path(self.sim_config.progress_path)
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------ lock

    def _acquire_lock(self) -> bool:
        try:
            fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except OSError:
            return False

    def _release_lock(self) -> None:
        try:
            self.lock_file.unlink()
        except OSError:
            pass

    # -------------------------------------------------------------- process ops

    def _is_alive(self, pid: int | None, started_at: str | None = None) -> bool:
        """Process liveness with PID-reuse protection (creation time check)."""

        if not pid:
            return False
        try:
            proc = psutil.Process(pid)
            if not proc.is_running():
                return False
            if started_at:
                try:
                    start = datetime.fromisoformat(started_at)
                    create = datetime.fromtimestamp(proc.create_time())
                    if create < start:
                        return False  # reused PID, not our process
                except (TypeError, ValueError):
                    pass
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    def _terminate(self, pid: int | None) -> None:
        if not pid:
            return
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            proc.terminate()
        except psutil.Error:
            pass

    def _kill(self, pid: int | None) -> None:
        if not pid:
            return
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            proc.kill()
        except psutil.Error:
            pass

    # ------------------------------------------------------------------ start

    def start(
        self,
        reset: bool = False,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        record = self._read_run()
        if record and self._is_alive(record.get("pid"), record.get("started_at")):
            raise RunConflictError(
                f"sim is already running (pid={record.get('pid')})"
            )
        if not self._acquire_lock():
            # Lock exists: only a live process may hold it; otherwise it is a
            # stale artifact from a crashed API and we may steal it.
            record = self._read_run()
            if record and self._is_alive(record.get("pid"), record.get("started_at")):
                raise RunConflictError("sim run lock is held by a live process")
            self._release_lock()
            if not self._acquire_lock():
                raise RunConflictError("could not acquire the sim run lock")

        try:
            return self._spawn(reset, start_date, end_date)
        except Exception:
            self._release_lock()
            raise

    def _spawn(
        self,
        reset: bool,
        start_date: str | None,
        end_date: str | None,
    ) -> dict:
        argv = list(self.cmd_builder())
        if start_date:
            argv += ["--start", start_date]
        if end_date:
            argv += ["--end", end_date]

        # A leftover STOP/STOPPED signal or stale progress would make the new
        # run stop instantly or show a wrong phase; clear both.
        self._clear_stop_signal()
        self._remove(self.sim_config.progress_path if self.sim_config else None)

        log_path = self.root / "logs" / (
            f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

        popen_kwargs: dict = {}
        if sys.platform == "win32":
            # Detach from the API console so Ctrl+C on uvicorn never reaches
            # the simulation process.
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        with open(log_path, "a", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **popen_kwargs,
            )

        record = {
            "status": "starting",
            "pid": proc.pid,
            "started_at": _now_iso(),
            "stopping_at": None,
            "ended_at": None,
            "exit_code": None,
            "error": None,
            "reset": reset,
            "start_date": start_date,
            "end_date": end_date,
            "args": argv,
            "log_path": str(log_path),
        }
        self._write_run(record)
        logger.info(f"Simulation started: pid={proc.pid} args={argv}")
        return self.status()

    # ----------------------------------------------------------------- status

    def status(self) -> dict:
        record = self._read_run()
        progress = self._progress()
        base = {
            "state": "idle",
            "pid": None,
            "started_at": None,
            "stopping_at": None,
            "ended_at": None,
            "exit_code": None,
            "error": None,
            "reset": False,
            "start_date": None,
            "end_date": None,
            "log_path": None,
            "phase": progress.get("phase"),
            "current_date": progress.get("current_date"),
            "equity": progress.get("equity"),
            "progress_updated_at": progress.get("updated_at"),
        }
        if not record:
            return base

        pid = record.get("pid")
        started_at = record.get("started_at")
        alive = self._is_alive(pid, started_at)
        state = record.get("status", "idle")

        if alive:
            if state == "starting":
                # The runner writes phase=loading during data/factor warm-up;
                # anything past that means the day loop is running.
                phase = progress.get("phase")
                if phase and phase != "loading":
                    record = dict(record)
                    record["status"] = "running"
                    self._write_run(record)
                    state = "running"
            if state == "stopping":
                self._maybe_escalate(record)
                record = self._read_run() or record
                state = record.get("status", "stopping")
        else:
            if state not in TERMINAL_STATES:
                record = dict(record)
                if state == "stopping":
                    record["status"] = "stopped"
                else:
                    phase = progress.get("phase")
                    if phase == "finished":
                        record["status"] = "finished"
                    elif phase == "stopped":
                        record["status"] = "stopped"
                    elif phase == "error":
                        record["status"] = "error"
                    else:
                        record["status"] = "error"
                        record["error"] = (
                            "process exited without a terminal progress phase"
                        )
                record["ended_at"] = _now_iso()
                self._write_run(record)
            self._release_lock()
            state = record.get("status", state)

        return {
            **base,
            "state": state,
            "pid": pid,
            "started_at": record.get("started_at"),
            "stopping_at": record.get("stopping_at"),
            "ended_at": record.get("ended_at"),
            "exit_code": record.get("exit_code"),
            "error": record.get("error"),
            "reset": record.get("reset", False),
            "start_date": record.get("start_date"),
            "end_date": record.get("end_date"),
            "log_path": record.get("log_path"),
        }

    # ------------------------------------------------------------------- stop

    def stop(self) -> dict:
        record = self._read_run()
        if not record:
            return {"ok": True, "state": "idle", "message": "no run to stop"}
        if not self._is_alive(record.get("pid"), record.get("started_at")):
            # Dead process: reconcile the terminal state and report it.
            return {"ok": True, **self.status()}

        self._write_stop_signal()
        record = dict(record)
        record["status"] = "stopping"
        record["stopping_at"] = _now_iso()
        self._write_run(record)
        self._schedule_escalation(record.get("pid"))
        return {"ok": True, "state": "stopping", "pid": record.get("pid")}

    def _write_stop_signal(self) -> None:
        if self.sim_config is None:
            return
        try:
            Path(self.sim_config.stop_signal_path).write_text(
                "STOP", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(f"Could not write stop signal: {exc}")

    def _schedule_escalation(self, pid: int | None) -> None:
        """Background escalation so stop() returns immediately. If the API
        process is restarted, the next status() poll continues the escalation
        via ``_maybe_escalate``."""

        if self._escalation_thread and self._escalation_thread.is_alive():
            return
        thread = threading.Thread(
            target=self._escalate_after_grace, args=(pid,), daemon=True
        )
        thread.start()
        self._escalation_thread = thread

    def _escalate_after_grace(self, pid: int | None) -> None:
        deadline = time.monotonic() + self.stop_grace
        while time.monotonic() < deadline:
            time.sleep(1)
            if not self._is_alive(pid):
                return
        self._terminate(pid)
        deadline = time.monotonic() + self.kill_grace
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if not self._is_alive(pid):
                return
        self._kill(pid)

    def _maybe_escalate(self, record: dict) -> None:
        """Poll-time escalation fallback (API restarts lose the thread)."""

        try:
            elapsed = (
                datetime.now() - datetime.fromisoformat(record["stopping_at"])
            ).total_seconds()
        except (KeyError, TypeError, ValueError):
            return
        pid = record.get("pid")
        if elapsed > self.stop_grace + self.kill_grace:
            self._kill(pid)
        elif elapsed > self.stop_grace:
            self._terminate(pid)

    # ------------------------------------------------------------------ reset

    def reset(self) -> dict:
        """Archive the current state, then reset it.

        The archive (``scripts/archive_run.py --mode sim --commit``) must
        succeed first; a failed archive aborts the reset so no history is
        ever lost.
        """

        record = self._read_run()
        if record and self._is_alive(record.get("pid"), record.get("started_at")):
            raise RunConflictError("cannot reset while a sim run is active")
        if self.sim_config is None:
            return {"ok": False, "reason": "config load failed"}

        from ashare_trading.portfolio import SimulationPortfolio

        portfolio = SimulationPortfolio(
            self.sim_config.initial_capital, self.sim_config.state_path
        )
        if portfolio.has_history:
            ok, message = self._archive_run()
            if not ok:
                return {"ok": False, "reason": f"archive failed: {message}"}
            archive_note = message
        else:
            archive_note = "no history to archive"

        portfolio.reset()
        # Park the per-day paper trail so stale dates can never be mistaken
        # for the new run's output.
        self._backup_dir(self.sim_config.orders_dir)
        self._backup_dir(self.sim_config.trades_dir)
        self._remove(self.sim_config.progress_path)
        self._clear_stop_signal()
        logger.success(f"Simulation reset. {archive_note}")
        return {"ok": True, "archive": archive_note}

    def _archive_run(self) -> tuple[bool, str]:
        script = self.root / "scripts" / "archive_run.py"
        if not script.exists():
            return False, f"archive script not found: {script}"
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "--mode", "sim", "--commit"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "archive timed out"
        except OSError as exc:
            return False, str(exc)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        message = tail[-1] if tail else f"exit code {proc.returncode}"
        return proc.returncode == 0, message

    # ---------------------------------------------------------------- helpers

    def _clear_stop_signal(self) -> None:
        if self.sim_config is None:
            return
        try:
            path = Path(self.sim_config.stop_signal_path)
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning(f"Could not clear stop signal: {exc}")

    @staticmethod
    def _remove(path: Path | None) -> None:
        if path is None:
            return
        try:
            Path(path).unlink()
        except OSError:
            pass

    @staticmethod
    def _backup_dir(directory: Path) -> None:
        if not Path(directory).exists():
            return
        backup = Path(directory).with_name(
            Path(directory).name + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.replace(directory, backup)
