from __future__ import annotations

from pathlib import Path

from ashare_data.io_utils import atomic_write_json, read_json_safe


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
