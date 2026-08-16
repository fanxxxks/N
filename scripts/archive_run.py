"""Archive an experiment run (train / backtest / sim) into experiments/ for permanent traceability.

Goal: answer "which formula + which config + which code + which data -> what result" forever.

Snapshot layout (experiments/<YYYYMMDD>_<slug>/):
    manifest.json          run metadata: mode, git commit, dirty flag, formula, data end date,
                           SHA-256 of config/metrics/model, storage decisions
    formula.json           copy of the formula/strategy JSON (tokens + formula_text + best_reward)
    config.yaml            copy of the config file used for the run
    metrics_summary.json   extracted key metrics (always written, tiny)
    metrics.json           full metrics file (only when <= --max-metrics-size-mb)
    model.<ext>            model weights (only when <= --max-model-size-mb, default 2;
                           otherwise only the SHA-256 reference is recorded)

Usage:
    python scripts/archive_run.py --mode backtest
    python scripts/archive_run.py --mode train --commit
    python scripts/archive_run.py --mode sim
    python scripts/archive_run.py --mode manual --formula f.json --config c.yaml \
        --metrics m.json --model w.pt --name my_run

--commit commits only the new experiments/<dir> (never pushes, never touches other changes).
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parents[1]

# Default artifact locations (relative to repo root) for the standard pipeline modes.
DEFAULTS = {
    "train": {
        "formula": "data/best_ashare_strategy.json",
        "model": "data/ashare_model.pt",
        "config": "config/ashare_config.yaml",
    },
    "backtest": {
        "formula": "data/best_ashare_strategy.json",
        "metrics": "data/backtest_result.json",
        "config": "config/ashare_config.yaml",
    },
    "sim": {
        "formula": "data/best_ashare_strategy.json",  # optional for sim
        "metrics": "data/sim_portfolio_state.json",
        "config": "config/ashare_config.yaml",
    },
    "protocol": {
        "metrics": "data/protocol_result.json",
        "config": "config/ashare_config.yaml",
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_slug(text, max_len=48):
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len] or "run"


def git_commit_sha(root):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def git_is_dirty(root, exclude_prefix="experiments/"):
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, check=True,
        )
        for line in out.stdout.splitlines():
            path = line[3:].strip()
            if not path or path.startswith(exclude_prefix):
                continue
            return True
        return False
    except Exception:
        return None


def load_formula_info(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "formula": data.get("formula"),
        "formula_text": data.get("formula_text"),
        "best_reward": data.get("best_reward"),
        # Reward provenance: best_reward is only comparable within the same
        # reward implementation version (absent on legacy artifacts).
        "reward_version": data.get("reward_version"),
    }


def summarize_metrics(data):
    """Keep top-level scalars and a scalar-only copy of a nested 'metrics' dict."""
    summary = {}
    for key, value in data.items():
        if key == "metrics" and isinstance(value, dict):
            summary["metrics"] = {
                k: v for k, v in value.items()
                if isinstance(v, (int, float, str, bool))
            }
        elif isinstance(value, (int, float, str, bool)):
            summary[key] = value
    return summary


def summarize_protocol(data):
    """Protocol artifact summary: provenance scalars, per-candidate
    aggregates and the top trial.  The raw per-fold, per-seed rows stay in
    the archived metrics.json so later analysis can always drill down."""

    summary = {}
    for key in (
        "protocol_version",
        "reward_version",
        "frequency",
        "horizon",
        "tier",
        "steps",
        "batch_size",
        "seeds",
        "data_end_date",
        "n_candidates",
        "dsr",
        "max_t",
    ):
        if key in data:
            summary[key] = data[key]
    if "aggregates" in data:
        summary["aggregates"] = data["aggregates"]
    top = data.get("top_trial")
    if top:
        summary["top_trial"] = {
            k: top.get(k)
            for k in (
                "candidate",
                "formula_text",
                "fold_train_end",
                "fold_test_end",
                "seed",
                "sharpe",
            )
        }
    return summary


def data_end_date(root, config_path):
    """Best-effort: max trade_date in the local DuckDB.

    Table name and DB path come from the archived config (relative paths are
    resolved against the repo root), falling back to the standard local DB.
    """
    try:
        import duckdb
        import yaml

        table = "daily_bar"
        db_path = root / "data" / "ashare.duckdb"
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            table = cfg.get("daily_table", table)
            db_raw = cfg.get("duckdb_path")
            if db_raw:
                candidate = Path(db_raw)
                db_path = candidate if candidate.is_absolute() else root / candidate
        except Exception:
            pass
        if not db_path.exists():
            return None
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            row = con.execute("SELECT MAX(trade_date) FROM %s" % table).fetchone()
            return str(row[0]) if row and row[0] else None
        finally:
            con.close()
    except Exception:
        return None


def pick_run_dir(root, date_str, slug):
    """experiments/<date>_<slug>, with _2/_3 suffix on collision."""
    experiments = root / "experiments"
    base = "%s_%s" % (date_str, slug)
    candidate = experiments / base
    n = 2
    while candidate.exists():
        candidate = experiments / ("%s_%d" % (base, n))
        n += 1
    return candidate


def resolve_path(root, cli_value, default_rel):
    if cli_value:
        return Path(cli_value)
    if default_rel:
        path = root / default_rel
        if path.exists():
            return path
    return None


def load_effective_config(config_path, root):
    """Baseline YAML merged with the runtime overrides file (best effort)."""
    try:
        import sys as _sys

        # When invoked as `python scripts/archive_run.py`, sys.path[0] is the
        # scripts dir; make the package roots importable first.
        for candidate in (str(Path(root)), str(DEFAULT_ROOT)):
            if candidate not in _sys.path:
                _sys.path.insert(0, candidate)
        from ashare_data.config import load_config

        return load_config(config_path, project_root=root)
    except Exception:
        return None


def git_commit_run_dir(run_dir, mode, root):
    """Stage and commit only the archived run dir. Returns (ok, message)."""
    rel = run_dir.relative_to(root).as_posix()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", rel], cwd=root,
        capture_output=True, text=True,
    )
    if not status.stdout.strip():
        return False, "nothing to commit (dir already committed?)"
    subprocess.run(["git", "add", "--", rel], cwd=root, check=True)
    out = subprocess.run(
        ["git", "commit", "-m", "experiment(%s): %s" % (mode, run_dir.name), "--", rel],
        cwd=root, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return False, out.stderr.strip() or out.stdout.strip()
    return True, out.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Archive an experiment run into experiments/.")
    parser.add_argument("--mode", required=True,
                        choices=["train", "backtest", "sim", "manual", "protocol"],
                        help="run type; sets default artifact paths (manual = explicit paths only)")
    parser.add_argument("--formula", help="path to formula/strategy JSON")
    parser.add_argument("--config", help="path to config file (yaml or json)")
    parser.add_argument("--metrics", help="path to metrics JSON")
    parser.add_argument("--model", help="path to model weights")
    parser.add_argument("--name", help="slug override (default: sanitized formula_text)")
    parser.add_argument("--commit", action="store_true",
                        help="git commit the archived run dir after writing it")
    parser.add_argument("--dry-run", action="store_true", help="print plan, write nothing")
    parser.add_argument("--max-model-size-mb", type=float, default=2.0,
                        help="models larger than this are hash-referenced only (default 2)")
    parser.add_argument("--max-metrics-size-mb", type=float, default=5.0,
                        help="metrics files larger than this are summarized only (default 5)")
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="repo root (default: auto-detected; used by tests)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    defaults = DEFAULTS.get(args.mode, {})
    formula_path = resolve_path(root, args.formula, defaults.get("formula"))
    config_path = resolve_path(root, args.config, defaults.get("config"))
    metrics_path = resolve_path(root, args.metrics, defaults.get("metrics"))
    model_path = resolve_path(root, args.model, defaults.get("model"))

    if args.mode in ("train", "backtest") and formula_path is None:
        print("ERROR: formula file not found; provide --formula or generate one first", file=sys.stderr)
        return 2
    if not any([formula_path, config_path, metrics_path, model_path]):
        print("ERROR: no artifacts provided; nothing to archive", file=sys.stderr)
        return 2

    # Slug: --name > formula_text > protocol tier > generic.
    if args.name:
        slug = sanitize_slug(args.name)
    elif formula_path:
        info = load_formula_info(formula_path)
        slug = sanitize_slug(info.get("formula_text") or "run")
    elif args.mode == "protocol" and metrics_path and metrics_path.exists():
        try:
            tier = json.loads(metrics_path.read_text(encoding="utf-8")).get("tier")
            slug = sanitize_slug(f"protocol_{tier}" if tier else "protocol")
        except Exception:
            slug = "protocol"
    else:
        slug = "run"

    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = pick_run_dir(root, date_str, slug)

    if args.dry_run:
        print("[dry-run] would create: %s" % run_dir.relative_to(root))
        print("[dry-run] mode=%s formula=%s config=%s metrics=%s model=%s" % (
            args.mode,
            formula_path.relative_to(root) if formula_path and formula_path.is_relative_to(root) else formula_path,
            config_path.relative_to(root) if config_path and config_path.is_relative_to(root) else config_path,
            metrics_path.relative_to(root) if metrics_path and metrics_path.is_relative_to(root) else metrics_path,
            model_path.relative_to(root) if model_path and model_path.is_relative_to(root) else model_path,
        ))
        if args.commit:
            print("[dry-run] would git-commit experiments/ dir")
        return 0

    # Capture repo state before touching the tree (so the run dir itself isn't "dirty").
    commit_sha = git_commit_sha(root)
    dirty = git_is_dirty(root)

    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "slug": slug,
        "git_commit": commit_sha,
        "git_dirty": dirty,
        "formula": None,
        "config": None,
        "metrics": None,
        "model": None,
        "protocol": None,
        "data_end_date": None,
        "files": [],
    }
    files = []

    def record(name, path, size):
        files.append({"name": name, "bytes": size})
        manifest["files"].append({"name": name, "bytes": size})

    if formula_path:
        info = load_formula_info(formula_path)
        shutil.copyfile(formula_path, run_dir / "formula.json")
        record("formula.json", formula_path, formula_path.stat().st_size)
        manifest["formula"] = info

    if config_path:
        shutil.copyfile(config_path, run_dir / "config.yaml")
        record("config.yaml", config_path, config_path.stat().st_size)
        manifest["config"] = {"source": config_path.name, "sha256": sha256_file(config_path)}
        manifest["data_end_date"] = data_end_date(root, config_path)
        # Snapshot the *effective* config (baseline + runtime overrides) so
        # the archived run stays reproducible even after web-UI config edits.
        effective = load_effective_config(config_path, root)
        if effective:
            eff_path = run_dir / "config_effective.yaml"
            eff_path.write_text(
                yaml.safe_dump(effective, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            record("config_effective.yaml", eff_path, eff_path.stat().st_size)
            manifest["config"]["effective_sha256"] = sha256_file(eff_path)

    if metrics_path:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        is_protocol = args.mode == "protocol" and "protocol_version" in data
        summary_path = run_dir / "metrics_summary.json"
        summary = summarize_protocol(data) if is_protocol else summarize_metrics(data)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record("metrics_summary.json", summary_path, summary_path.stat().st_size)
        entry = {"source": metrics_path.name, "sha256": sha256_file(metrics_path)}
        if metrics_path.stat().st_size <= args.max_metrics_size_mb * 1024 * 1024:
            shutil.copyfile(metrics_path, run_dir / "metrics.json")
            record("metrics.json", metrics_path, metrics_path.stat().st_size)
            entry["stored"] = True
        else:
            entry["stored"] = False
            entry["reason"] = "exceeds --max-metrics-size-mb (summary kept)"
        manifest["metrics"] = entry
        if is_protocol:
            manifest["protocol"] = {
                "version": data.get("protocol_version"),
                "frequency": data.get("frequency"),
                "horizon": data.get("horizon"),
                "tier": data.get("tier"),
                "steps": data.get("steps"),
                "batch_size": data.get("batch_size"),
                "n_folds": len(data.get("folds", [])),
                "n_seeds": len(data.get("seeds", [])),
                "n_candidates": data.get("n_candidates"),
                "dsr": data.get("dsr"),
                "max_t": data.get("max_t"),
                "top_candidate": (data.get("top_trial") or {}).get("candidate"),
            }

    if model_path:
        model_hash = sha256_file(model_path)
        model_dest = run_dir / ("model" + model_path.suffix)
        entry = {
            "source": model_path.name,
            "sha256": model_hash,
            "size_bytes": model_path.stat().st_size,
        }
        if model_path.stat().st_size <= args.max_model_size_mb * 1024 * 1024:
            shutil.copyfile(model_path, model_dest)
            record(model_dest.name, model_path, model_path.stat().st_size)
            entry["stored"] = True
        else:
            entry["stored"] = False
            entry["reason"] = "exceeds --max-model-size-mb (SHA-256 reference only)"
        manifest["model"] = entry

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    record("manifest.json", manifest_path, manifest_path.stat().st_size)

    rel = run_dir.relative_to(root).as_posix()
    print("archived: %s" % rel)
    print("  formula: %s" % (manifest["formula"] or {}).get("formula_text"))
    print("  git: %s (dirty=%s)" % (commit_sha, dirty))
    print("  data_end_date: %s" % manifest["data_end_date"])
    if manifest["model"]:
        print("  model: stored=%s sha256=%s" % (manifest["model"]["stored"], manifest["model"]["sha256"][:12]))

    if args.commit:
        ok, msg = git_commit_run_dir(run_dir, args.mode, root)
        print(("committed: " + msg.splitlines()[-1]) if ok else "commit skipped: " + msg)
        return 0 if ok else 1
    else:
        print("hint: re-run with --commit to commit this run dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
