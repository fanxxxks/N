from __future__ import annotations

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
