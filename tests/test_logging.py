from __future__ import annotations

import dataclasses
import re
import threading
import time
from pathlib import Path

from loguru import logger

from ashare_logging import export_log_txt, get_log_text, setup_run_logging


def test_run_logging_captures_and_exports(tmp_path: Path):
    setup_run_logging(log_dir=tmp_path, run_name="smoke", reset=True)
    logger.info("hello logging")
    text = get_log_text()
    assert "hello logging" in text

    out = export_log_txt(path=tmp_path / "run.txt")
    content = out.read_text(encoding="utf-8")
    assert "hello logging" in content
    assert "Logs exported" in get_log_text()


# ---------------------------------------------------------------------------
# IP-11 (03-F-07): critical-path run-identity header helpers.  The header is
# emitted through the single loguru pipeline (no second telemetry path), one
# fixed-format INFO line per AGENTS §4.5: run_id + git commit + config
# sha256 + the run's key version set.
# ---------------------------------------------------------------------------


def test_new_log_run_id_is_uuid_hex_and_unique():
    # Imported lazily so the pre-implementation RED failure is the missing
    # helper, not a collection error that would mask the whole module.
    from ashare_logging import new_log_run_id

    first = new_log_run_id()
    second = new_log_run_id()
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert first != second


def test_git_commit_returns_hex_in_repo_and_none_outside(tmp_path: Path):
    from ashare_logging import git_commit

    commit = git_commit()  # the test session runs inside the repository
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert git_commit(tmp_path) is None  # tmp_path is not a git repository


def test_canonical_config_sha256_is_order_independent_and_sensitive():
    import hashlib
    import json

    from ashare_logging import canonical_config_sha256

    @dataclasses.dataclass
    class _Cfg:
        alpha: int = 1
        data_dir: Path = Path("data")

    expected = hashlib.sha256(
        json.dumps(
            {"alpha": 1, "data_dir": "data"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    # Dataclass and equivalent mapping hash identically (asdict + str(Path)).
    assert canonical_config_sha256(_Cfg()) == expected
    assert canonical_config_sha256({"data_dir": "data", "alpha": 1}) == expected
    # Any value change moves the hash.
    assert canonical_config_sha256({"alpha": 2, "data_dir": "data"}) != expected


def test_emit_run_identity_fixed_format_single_pipeline(tmp_path: Path):
    from ashare_logging import emit_run_identity

    setup_run_logging(log_dir=tmp_path, run_name="identity", reset=True)
    line = emit_run_identity(
        run_id="a" * 32,
        config_sha256="b" * 64,
        versions={"zeta": "2", "alpha": "1"},
        commit="c" * 40,
    )
    assert line == (
        f"run identity: run_id={'a' * 32} git_commit={'c' * 40} "
        f"config_sha256={'b' * 64} "
        'versions={"alpha":"1","zeta":"2"}'
    )
    # One pipeline: the line is visible in the standard memory sink.
    assert line in get_log_text()


def test_emit_run_identity_renders_unknown_and_none_defaults(tmp_path: Path):
    from ashare_logging import emit_run_identity

    setup_run_logging(log_dir=tmp_path, run_name="identity2", reset=True)
    line = emit_run_identity(run_id="f" * 32)
    assert re.fullmatch(
        r"run identity: run_id=f{32} git_commit=([0-9a-f]{40}|unknown) "
        r"config_sha256=none versions=\{\}",
        line,
    ), line


def test_exit_guard_noop_without_survivors():
    # Main thread only: the guard is a silent no-op.
    from ashare_logging import guard_process_exit

    guard_process_exit(timeout=0.1)


def test_exit_guard_joins_threads_that_finish_in_time():
    from ashare_logging import guard_process_exit

    done = threading.Event()

    def worker():
        time.sleep(0.05)
        done.set()

    thread = threading.Thread(target=worker, name="guard-finishes")
    thread.start()
    guard_process_exit(timeout=10.0)
    assert done.is_set()
    assert not thread.is_alive()


def test_exit_guard_forces_loud_exit_after_timeout(monkeypatch, tmp_path):
    """F-08 (sync entry, force_exit=True): a surviving non-daemon thread
    must never hang the interpreter silently — its stack is logged at
    ERROR and the exit is forced loudly after the bounded join."""
    from ashare_logging import guard_process_exit

    release = threading.Event()
    started = threading.Event()

    def hang():
        started.set()
        release.wait(30)

    thread = threading.Thread(target=hang, name="guard-hangs")
    thread.start()
    assert started.wait(5)
    setup_run_logging(log_dir=tmp_path, run_name="guard", reset=True)
    forced: list[int] = []
    monkeypatch.setattr("os._exit", lambda code=0: forced.append(code))
    try:
        guard_process_exit(timeout=0.2)
    finally:
        release.set()
        thread.join(5)
    assert forced == [3]
    text = get_log_text()
    assert "guard-hangs" in text  # survivor named in the ERROR report
    assert "forcing process exit" in text
    assert "in hang" in text  # the survivor's stack is the evidence


def test_exit_guard_detect_only_never_forces(monkeypatch, tmp_path):
    """F-08 (pytest entry, force_exit=False): the session finalizer must
    name a surviving non-daemon thread in the (immediately exported) log
    but must NOT force-exit — the terminal report and CI result are
    published after teardown and a forced exit would swallow them."""
    from ashare_logging import guard_process_exit

    release = threading.Event()
    started = threading.Event()

    def hang():
        started.set()
        release.wait(30)

    thread = threading.Thread(target=hang, name="guard-detect-only")
    thread.start()
    assert started.wait(5)
    setup_run_logging(log_dir=tmp_path, run_name="guard2", reset=True)
    forced: list[int] = []
    monkeypatch.setattr("os._exit", lambda code=0: forced.append(code))
    try:
        guard_process_exit(timeout=0.2, force_exit=False)
    finally:
        release.set()
        thread.join(5)
    assert forced == []  # detect-only: the forced path never fires
    text = get_log_text()
    assert "guard-detect-only" in text  # survivor named with its stack
    assert "in hang" in text
    assert "exit may lag" in text  # explicit, bounded, non-fatal warning
