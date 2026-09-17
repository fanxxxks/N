"""Read-only D7 data-eligibility gate CLI for the R1 campaign (p21 §6.2).

Resolves one experiment config's preregistered feature set through the
single authorities (``ashare_domain`` registry + ``ashare_model.data_tier``
+ ``vocab.deprecated_names`` — the same functions the protocol tier
restriction consumes, never a second mapping) and runs the p17 §12
per-formula-dependency eligibility check against the real stores.

Read-only by construction: opens both databases read-only and writes only
the requested verdict JSON (``--output``).  No research conclusion; the
verdict is a run precondition (p21 §6.2), not evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import duckdb

from ashare_data.config import load_config, make_data_config
from ashare_domain import RESEARCH_DOMAINS, resolve_domain
from ashare_model.data_eligibility import check_runspec_data_eligibility
from ashare_model.data_tier import feature_tier
from ashare_model.runspec import WindowSpec
from ashare_model.vocab import FORMULA_VOCAB

_TIER_SETS = {"A": ("A",), "AB": ("A", "B")}


def resolve_r1_feature_set(domain_id: str, tier_set: str | None) -> list[str]:
    """The p21 §4 preregistered feature-set rule, via the single authorities.

    Domain features (registry) ∩ tier set (``feature_tier``) minus
    deprecated names (never sampleable, grammar v4+ rule).  ``unified``
    yields every samplable feature; ``tier_set`` must be None there (the
    p21 configs never combine them).
    """

    deprecated = FORMULA_VOCAB.deprecated_names
    allowed_tiers = _TIER_SETS[tier_set] if tier_set else None
    if domain_id == "unified":
        if tier_set is not None:
            raise ValueError("tier_set requires a research domain, not unified")
        return [
            name
            for name in FORMULA_VOCAB.feature_names
            if name not in deprecated
        ]
    domain = resolve_domain(domain_id)
    out: list[str] = []
    for name in FORMULA_VOCAB.feature_names:
        if name in deprecated or name not in domain.features:
            continue
        if allowed_tiers is not None and feature_tier(name).value not in allowed_tiers:
            continue
        out.append(name)
    return out


def universe_codes_from_legacy(db_path: str, table: str) -> list[str]:
    """PIT-eligible codes: the legacy constituents interval table."""

    con = duckdb.connect(db_path, read_only=True)
    try:
        # SQL-injection hardening: the table name goes through the bound
        # query_table(?) parameter instead of f-string interpolation.
        rows = con.execute(
            "SELECT DISTINCT ts_code FROM query_table(?)", [table]
        ).fetchall()
    finally:
        con.close()
    return sorted(str(r[0]) for r in rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--tier-set", choices=sorted(_TIER_SETS), default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    raw = load_config(args.config, project_root=Path(__file__).resolve().parents[1])
    config = make_data_config(raw, Path(__file__).resolve().parents[1])
    features = resolve_r1_feature_set(args.domain, args.tier_set)
    codes = universe_codes_from_legacy(config.duckdb_path, config.constituents_table)

    class _SpecShim:
        """The eligibility check reads only dev_window.{start,end} (d8) for
        the financial coverage window and spec_id for the verdict payload.
        Mirrors ``resolve_runtime_runspec``: dev window = (data start,
        backtest.train_end_date).  This CLI is a read-only probe, not a
        formal RunSpec owner."""

        spec_id = f"r1-gate-probe:{Path(args.config).stem}"
        dev_window = WindowSpec(
            start=str(raw["start_date"]).replace("-", ""),
            end=str(raw["backtest"]["train_end_date"]).replace("-", ""),
        )

    verdict = check_runspec_data_eligibility(
        _SpecShim(), features, codes, config
    )
    payload = {
        "config": args.config,
        "domain": args.domain,
        "tier_set": args.tier_set,
        "n_features": len(features),
        "features": features,
        "n_universe_codes": len(codes),
        "ok": verdict.ok,
        "checks": {k: asdict(v) for k, v in verdict.checks.items()},
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
