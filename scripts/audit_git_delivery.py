#!/usr/bin/env python3
"""Audit indexed blobs and staged paths; findings contain paths, never values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"(^|/)(?:\.mimosa|\.zcode|\.agent-teams|\.venv|venv|__pycache__|node_modules|dist|tmp|temp|factor_cache)(/|$)|"
    r"(^|/)(?:nul|id_rsa|id_ed25519)$|\.(?:duckdb(?:\.wal)?|sqlite(?:3)?|db|pt|pth|pem|key|pyc|log)$", re.I)
SECRETS = [re.compile(pattern.encode()) for pattern in (
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\b(?:ghp_|github_pat_|gho_|ghu_|ghs_)[A-Za-z0-9_]{30,}\b",
    r"\bAKIA[0-9A-Z]{16}\b", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b",
    r"https?://[^\s/:]+:[^\s/@]+@[A-Za-z0-9.-]+",
    r"(?im)^\s*(?:TUSHARE_TOKEN|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|SSH_PRIVATE_KEY)\s*=\s*['\"]?[A-Za-z0-9_+/=-]{24,}")]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def indexed_blobs(names):
    queries = "".join(":" + name + "\n" for name in names).encode()
    raw = subprocess.check_output(["git", "cat-file", "--batch"], input=queries, cwd=ROOT)
    offset = 0
    for name in names:
        end = raw.index(b"\n", offset)
        header = raw[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("index blob unavailable: " + name)
        size = int(header[2])
        offset = end + 1
        yield name, raw[offset:offset + size]
        offset += size + 1


def audit():
    names = [n for n in git("ls-files", "-z").decode().split("\0") if n]
    staged = [n for n in git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0") if n]
    result = {"scope": "all current index blobs (existing tracked and staged), plus staged path list; not Git history",
        "indexed_file_count": len(names), "staged_paths": staged, "large_paths": [],
        "forbidden_paths": [], "credential_pattern_paths": []}
    for name, blob in indexed_blobs(names):
        example = name.endswith(".env.example")
        if FORBIDDEN.search(name) or (Path(name).name.startswith(".env") and not example):
            result["forbidden_paths"].append(name)
        if len(blob) > 1024 * 1024:
            result["large_paths"].append(name)
        if any(pattern.search(blob) for pattern in SECRETS):
            result["credential_pattern_paths"].append(name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return int(bool(report["forbidden_paths"] or report["credential_pattern_paths"] or report["large_paths"]))


if __name__ == "__main__":
    raise SystemExit(main())
