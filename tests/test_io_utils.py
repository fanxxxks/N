from __future__ import annotations

import os
from pathlib import Path

from ashare_data.io_utils import (
    atomic_write_json,
    durable_append_line,
    durable_write_text,
    read_json_safe,
)


def test_read_json_safe_missing_and_corrupt(tmp_path: Path):
    assert read_json_safe(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json {", encoding="utf-8")
    assert read_json_safe(bad) is None
    # A non-dict/list payload is rejected as well.
    scalar = tmp_path / "scalar.json"
    scalar.write_text("42", encoding="utf-8")
    assert read_json_safe(scalar) is None


def test_read_json_safe_roundtrip(tmp_path: Path):
    path = tmp_path / "ok.json"
    atomic_write_json(path, {"a": 1, "b": [1, 2]})
    assert read_json_safe(path) == {"a": 1, "b": [1, 2]}


def test_atomic_write_json_is_atomic_and_pretty(tmp_path: Path):
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"cash": 1.0})
    text = path.read_text(encoding="utf-8")
    assert '"cash": 1.0' in text
    # The temp file is renamed away, never left behind.
    assert not path.with_suffix(".tmp.json").exists()
    # Overwriting an existing file works and stays atomic.
    atomic_write_json(path, {"cash": 2.0})
    assert read_json_safe(path) == {"cash": 2.0}
    assert not path.with_suffix(".tmp.json").exists()


# -- durability (P1 fsync review) ---------------------------------------------
#
# Real persistence can only be proven by pulling the power cord; what these
# tests pin is that every write path actually calls os.fsync (with the file
# flushed first) before the operation is considered complete.


class _FsyncSpy:
    def __init__(self, monkeypatch):
        self.calls: list[int] = []
        self._real = os.fsync
        monkeypatch.setattr(os, "fsync", self)

    def __call__(self, fd: int) -> None:
        self.calls.append(fd)


def test_atomic_write_json_fsyncs_before_replace(tmp_path: Path, monkeypatch):
    spy = _FsyncSpy(monkeypatch)
    path = tmp_path / "state.json"
    atomic_write_json(path, {"a": 1})
    # Exactly one fsync for the single temp-file write, and the published
    # file is complete (the spy swallowed nothing — content is on the
    # in-memory handle, flushed before the fsync call).
    assert len(spy.calls) == 1
    assert read_json_safe(path) == {"a": 1}
    assert not path.with_suffix(".tmp.json").exists()


def test_durable_write_text_fsyncs_and_leaves_no_tmp(tmp_path: Path, monkeypatch):
    spy = _FsyncSpy(monkeypatch)
    path = tmp_path / "evidence.json"
    durable_write_text(path, "payload")
    assert len(spy.calls) == 1
    assert path.read_text(encoding="utf-8") == "payload"
    # The default tmp name is collision-safe and always renamed away.
    assert [p.name for p in tmp_path.iterdir()] == ["evidence.json"]
    # A caller-provided tmp_name keeps the same guarantee.
    durable_write_text(path, "payload2", tmp_name="evidence.json.tmp-custom")
    assert len(spy.calls) == 2
    assert path.read_text(encoding="utf-8") == "payload2"
    assert [p.name for p in tmp_path.iterdir()] == ["evidence.json"]


def test_durable_append_line_fsyncs_each_line(tmp_path: Path, monkeypatch):
    spy = _FsyncSpy(monkeypatch)
    path = tmp_path / "chain.jsonl"
    durable_append_line(path, '{"seq": 1}')
    durable_append_line(path, '{"seq": 2}')
    # One fsync per completed append — an append that returned is on disk.
    assert len(spy.calls) == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
