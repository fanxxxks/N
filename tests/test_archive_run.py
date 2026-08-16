"""Tests for scripts/archive_run.py (experiment archival mechanism)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ARCHIVE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_run.py"


def run_archive(root, *args):
    return subprocess.run(
        [sys.executable, str(ARCHIVE_SCRIPT), "--root", str(root), *args],
        cwd=root, capture_output=True, text=True,
    )


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path):
    """A minimal git repo with fake train/backtest artifacts."""
    assert git(tmp_path, "init", "-b", "main").returncode == 0
    git(tmp_path, "config", "user.name", "tester")
    git(tmp_path, "config", "user.email", "tester@example.com")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "data").mkdir()
    formula = {
        "formula": [8, 0, 0],
        "formula_text": "TURNOVER_CHG MUL RET_5",
        "best_reward": 0.123,
        "reward_version": "1",
    }
    (tmp_path / "data" / "best.json").write_text(json.dumps(formula), encoding="utf-8")
    (tmp_path / "config.yaml").write_text("model:\n  d_model: 64\n", encoding="utf-8")
    metrics = {
        "formula_text": "TURNOVER_CHG MUL RET_5",
        "metrics": {"sharpe": 1.5, "max_drawdown": 0.2},
        "dates": ["20240101", "20240102"],
        "equity": [1.0, 1.02],
    }
    (tmp_path / "data" / "result.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "data" / "model_small.pt").write_bytes(b"weights" * 200)
    (tmp_path / "data" / "model_big.pt").write_bytes(b"0" * (3 * 1024 * 1024))
    assert git(tmp_path, "add", ".").returncode == 0
    assert git(tmp_path, "commit", "-m", "init").returncode == 0
    return tmp_path


def test_manual_archive_records_everything(repo):
    r = run_archive(
        repo, "--mode", "manual",
        "--formula", "data/best.json",
        "--config", "config.yaml",
        "--metrics", "data/result.json",
        "--model", "data/model_small.pt",
        "--name", "demo",
    )
    assert r.returncode == 0, r.stderr
    exp_dirs = list((repo / "experiments").iterdir())
    assert len(exp_dirs) == 1
    run_dir = exp_dirs[0]
    assert run_dir.name.endswith("demo")
    assert run_dir.name[:8].isdigit()  # YYYYMMDD prefix

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "manual"
    assert manifest["formula"]["formula_text"] == "TURNOVER_CHG MUL RET_5"
    assert manifest["formula"]["best_reward"] == 0.123
    assert manifest["formula"]["reward_version"] == "1"
    assert manifest["config"]["sha256"] == hashlib.sha256(
        (repo / "config.yaml").read_bytes()).hexdigest()
    assert manifest["metrics"]["stored"] is True
    assert manifest["model"]["stored"] is True
    assert manifest["model"]["sha256"] == hashlib.sha256(
        (repo / "data" / "model_small.pt").read_bytes()).hexdigest()
    assert manifest["git_commit"] == git(repo, "rev-parse", "HEAD").stdout.strip()
    assert manifest["git_dirty"] is False

    for fname in ("formula.json", "config.yaml", "metrics.json",
                  "metrics_summary.json", "model.pt", "manifest.json"):
        assert (run_dir / fname).exists(), fname

    summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["sharpe"] == 1.5
    assert "dates" not in summary  # list values are dropped from the summary


def test_legacy_formula_without_reward_version_archives(repo):
    (repo / "data" / "best.json").write_text(
        json.dumps({"formula": [1, 2], "formula_text": "OLD", "best_reward": 0.5}),
        encoding="utf-8",
    )
    r = run_archive(repo, "--mode", "manual", "--formula", "data/best.json")
    assert r.returncode == 0, r.stderr
    run_dir = next((repo / "experiments").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["formula"]["reward_version"] is None


def test_big_model_is_hash_referenced_only(repo):
    r = run_archive(repo, "--mode", "manual", "--model", "data/model_big.pt", "--name", "big")
    assert r.returncode == 0, r.stderr
    run_dir = next((repo / "experiments").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["stored"] is False
    assert manifest["model"]["size_bytes"] == 3 * 1024 * 1024
    assert not (run_dir / "model.pt").exists()


def test_formula_derived_slug_is_sanitized(repo):
    r = run_archive(repo, "--mode", "manual", "--formula", "data/best.json")
    assert r.returncode == 0, r.stderr
    name = next((repo / "experiments").iterdir()).name
    assert name.endswith("TURNOVER_CHG_MUL_RET_5")


def test_run_dir_collision_gets_suffix(repo):
    for _ in range(2):
        r = run_archive(repo, "--mode", "manual", "--formula", "data/best.json", "--name", "same")
        assert r.returncode == 0, r.stderr
    names = sorted(p.name for p in (repo / "experiments").iterdir())
    assert names[0].endswith("same")
    assert names[1].endswith("same_2")


def test_commit_flag_commits_only_run_dir(repo):
    r = run_archive(repo, "--mode", "manual", "--formula", "data/best.json",
                    "--name", "cmt", "--commit")
    assert r.returncode == 0, r.stderr
    log = git(repo, "log", "--oneline").stdout
    assert "experiment(manual):" in log
    status = git(repo, "status", "--porcelain").stdout
    assert status.strip() == ""


def test_dry_run_writes_nothing(repo):
    r = run_archive(repo, "--mode", "manual", "--formula", "data/best.json", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "[dry-run]" in r.stdout
    assert not (repo / "experiments").exists()


def test_backtest_mode_requires_formula(repo):
    (repo / "data" / "best.json").unlink()
    r = run_archive(repo, "--mode", "backtest")
    assert r.returncode == 2


def test_archive_snapshots_effective_config(repo):
    """config_effective.yaml records the baseline merged with runtime
    overrides, even when the script runs as `python scripts/archive_run.py`."""

    (repo / "runtime_overrides.yaml").write_text(
        "sim:\n  max_positions: 10\n", encoding="utf-8"
    )
    r = run_archive(repo, "--mode", "manual", "--config", "config.yaml", "--name", "eff")
    assert r.returncode == 0, r.stderr
    run_dir = next((repo / "experiments").iterdir())
    eff_path = run_dir / "config_effective.yaml"
    assert eff_path.exists()
    eff = yaml.safe_load(eff_path.read_text(encoding="utf-8"))
    assert eff["model"]["d_model"] == 64  # baseline kept
    assert eff["sim"]["max_positions"] == 10  # override applied
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["effective_sha256"] == hashlib.sha256(
        eff_path.read_bytes()
    ).hexdigest()


# --- protocol mode ----------------------------------------------------------


def _write_protocol(repo):
    proto = {
        "protocol_version": "1",
        "reward_version": "1",
        "frequency": "daily",
        "horizon": 1,
        "tier": "screening",
        "steps": 50,
        "batch_size": 256,
        "seeds": [42, 7],
        "folds": [{"train_end": "2020-12-31", "test_end": "2021-12-31"}],
        "data_end_date": "20250801",
        "n_candidates": 4,
        "rows": [
            {
                "candidate": "trained",
                "sharpe": 1.2,
                "fold_test_end": "2021-12-31",
                "seed": 42,
                "val_reward": 0.8,
            }
        ],
        "aggregates": {
            "trained": {
                "n_rows": 1,
                "metrics": {"sharpe": {"median": 1.2}},
            }
        },
        "top_trial": {"candidate": "trained", "sharpe": 1.2, "seed": 42},
        "dsr": None,
        "max_t": None,
    }
    (repo / "data" / "protocol_result.json").write_text(
        json.dumps(proto), encoding="utf-8"
    )


def test_protocol_mode_archives_with_manifest_block(repo):
    _write_protocol(repo)
    r = run_archive(repo, "--mode", "protocol")
    assert r.returncode == 0, r.stderr
    run_dir = next((repo / "experiments").iterdir())
    assert run_dir.name.endswith("protocol_screening")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["protocol"]
    assert block["version"] == "1"
    assert block["frequency"] == "daily" and block["horizon"] == 1
    assert block["tier"] == "screening"
    assert block["steps"] == 50 and block["batch_size"] == 256
    assert block["n_folds"] == 1 and block["n_seeds"] == 2
    assert block["n_candidates"] == 4
    assert block["top_candidate"] == "trained"
    assert block["dsr"] is None and block["max_t"] is None
    assert manifest["metrics"]["stored"] is True

    summary = json.loads(
        (run_dir / "metrics_summary.json").read_text(encoding="utf-8")
    )
    assert summary["protocol_version"] == "1"
    assert summary["reward_version"] == "1"
    assert summary["aggregates"]["trained"]["n_rows"] == 1
    assert summary["top_trial"]["candidate"] == "trained"
    assert "rows" not in summary  # raw rows stay in the archived metrics.json

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rows"][0]["candidate"] == "trained"
    assert metrics["rows"][0]["val_reward"] == 0.8  # per-fold raw data preserved


def test_protocol_mode_does_not_require_formula(repo):
    (repo / "data" / "best.json").unlink()
    _write_protocol(repo)
    r = run_archive(repo, "--mode", "protocol")
    assert r.returncode == 0, r.stderr


def test_manual_mode_with_protocol_metrics_keeps_old_summary(repo):
    """A protocol-shaped metrics file archived as --mode manual must not
    trigger the protocol block (mode is the discriminator)."""
    _write_protocol(repo)
    r = run_archive(repo, "--mode", "manual", "--metrics", "data/protocol_result.json")
    assert r.returncode == 0, r.stderr
    run_dir = next((repo / "experiments").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] is None


def test_data_end_date_reads_duckdb_path_from_config(repo):
    """The archived config's duckdb_path (not the hardcoded local DB) is the
    source of data_end_date, so synthetic runs record their own database."""
    import duckdb

    db_path = repo / "data" / "custom.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE daily_bar (trade_date VARCHAR)")
    con.execute("INSERT INTO daily_bar VALUES ('20240801'), ('20240802')")
    con.close()
    (repo / "config_db.yaml").write_text(
        "duckdb_path: data/custom.duckdb\ndaily_table: daily_bar\n",
        encoding="utf-8",
    )
    r = run_archive(repo, "--mode", "manual", "--config", "config_db.yaml", "--name", "dbend")
    assert r.returncode == 0, r.stderr
    run_dir = next((repo / "experiments").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_end_date"] == "20240802"
