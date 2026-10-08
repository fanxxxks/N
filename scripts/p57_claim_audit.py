"""P57 WP18 claim audit — a PURE TEXT SCAN over the wp18 products
(收口需求 R2b; zero backtests, zero measurement rows).

Mechanical implementation of the WP17 wording contract as applied to
WP18's outputs.  A WP18 conclusion is DESCRIPTIVE development
validation; per FROZEN_17 (reports/wp17/FROZEN_17.md) the only legal
positive significance wording is the qualified template (§3.4) —
unreachable for WP18 because no ClaimRecord / max-t qualification was
run (the registered standing state) — and the WP18 report footer fixes
the required framing: 开发验证描述性结果、机械筛选门槛为研发资源分配门槛、
非统计显著性证明也非市场规律、不可作为名义 5% 显著性或样本外推断发布.

PATTERN WORD LISTS (declared here for auditability — every entry cites
its contract basis; append-only discipline: additions allowed at the
tail with a dated note, removals/rewordings forbidden):

Inferential patterns, tier "strong" (an UNMARKED hit defaults to
suspected_inferential — direct claim forms):
  nominal_5          nominal-5% / 名义 5%            FROZEN_17 §3.3/§3.4
  size_supported     "size supported"                FROZEN_17 §4.1 (replaced)
  fpr_at_most        误报率不超过 / at most          FROZEN_17 §4.3 (forbidden)
  p_threshold        p(值)? [<=≤] .05                FROZEN_17 §3.3 string C
  directed_sig       statistically significant /     FROZEN_17 §4.1 + WP18
                     显著(优于|超过|为正|…)          footer (directed claims)
  market_law         市场规律                        WP18 footer (banned)

Inferential patterns, tier "ambient" (an UNMARKED hit defaults to
needs_annotation — the words recur inside compliant disclosures):
  significance_en    significan(t|ce)                FROZEN_17 §3.4 surface
  significance_zh    显著                            same, Chinese
  oos                out-of-sample / 样本外          WP18 footer
  promotion_en       promot(e|es|ed|ion)             FROZEN_17 §3.4
  promotion_zh       晋级                            same, Chinese
  inference          inferen(ce|tial) / 推断         FROZEN_17 §3.4/§3.3
  proof              证明 / prov(e|es|en|ing)        WP18 footer (非…证明)

Compliance markers (any marker inside the ±160-char context window of
a hit keeps the hit descriptive_compliant):
  no_reliable       no reliable inference / 无法推断 / lacks calibrated
  not_statistical   非统计显著性证明 / 非市场规律 / not a statistical
  resource_gate     研发资源分配门槛
  forbidden_to_publish  不可作为 / 不得作为 / 不得支持 / not a verification
  negative_verb     未生成 / 未运行 / 未触发 / 未启用 / did not run
  descriptive       描述性 / descriptive / 开发验证 / development validation
  deferred          deferred / 推迟 / known issue
  fail_closed       qualified=False / not_qualified / EVIDENCE_NOT_REGISTERED
  measurement_inc   measurement_incomplete
  mechanical        机械筛选 / mechanical
  no_promotion      no promotion / 不晋级 / 未晋级 / cannot support promotion

A hit whose enclosing quoted token is a JSON object KEY (``"x":``) is
classified descriptive_compliant (json_field_name): a mechanical field
name records a value; the interpretive pathway is the claim-record
qualification machinery, which WP18 products do not carry.

DISPOSITIONS (append-only): a suspected_inferential hit is "handled"
when mapped here to the errata entry that records its suggested
disposition.  The corpus is NEVER rewritten to satisfy the audit.

Scope: reports/wp18/**  *.md *.json *.jsonl, EXCLUDING this package's
own outputs (self/parallel artifacts, not WP18 products):
_closure_recon.md, claim_audit.json, closure_verifications.json,
audit_findings_p57.md, P57_CLOSURE_REPORT.md, WP18_CLOSURE.md (the
closure/process documents carry their own descriptive-framework footer
and postdate the frozen research corpus; scanning them would make the
on-disk-vs-fresh equality self-referentially unstable).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WP18 = ROOT / "reports" / "wp18"
OUTPUT = WP18 / "claim_audit.json"

WINDOW = 160

INFERENTIAL_PATTERNS: list[tuple[str, str, str, str]] = [
    # (pattern_id, regex, tier, basis)
    ("nominal_5", r"nominal[-\s]?5%|名义\s*5\s*%", "strong",
     "FROZEN_17 §3.3/§3.4 nominal-5% claim surface"),
    ("size_supported", r"size supported", "strong",
     "FROZEN_17 §4.1 replaced wording"),
    ("fpr_at_most", r"误报率不超过|误报率\s*≤|false[-\s]positive rate is at most",
     "strong", "FROZEN_17 §4.3 forbidden upper-bound verification"),
    ("p_threshold", r"\bp\s*(?:值)?\s*[<=≤]\s*0?\.0?5\b", "strong",
     "FROZEN_17 §3.3 string C p-threshold surface"),
    ("directed_sig",
     r"statistically significant|统计上显著|显著(?:优于|超过|为正|为负|跑赢|跑输|超额|改善|更强)",
     "strong", "directed significance claim (FROZEN_17 §4.1; WP18 footer)"),
    ("market_law", r"市场规律", "strong",
     "WP18 footer: 非市场规律 (market-law claims banned)"),
    ("significance_en", r"significan(?:t|ce)", "ambient",
     "FROZEN_17 §3.4 significance wording (English)"),
    ("significance_zh", r"显著", "ambient",
     "FROZEN_17 §3.4 significance wording (Chinese)"),
    ("oos", r"out[-\s]of[-\s]sample|样本外", "ambient",
     "WP18 footer: 样本外推断 not publishable"),
    ("promotion_en", r"promot(?:e|es|ed|ion)", "ambient",
     "FROZEN_17 §3.4 promotion-claim surface (English)"),
    ("promotion_zh", r"晋级", "ambient",
     "FROZEN_17 §3.4 promotion-claim surface (Chinese)"),
    ("inference", r"inferen(?:ce|tial)|推断", "ambient",
     "FROZEN_17 §3.4 inference wording"),
    ("proof", r"证明|\bprov(?:e|es|en|ing)\b", "ambient",
     "WP18 footer: 非…证明 (proof wording; \\b keeps the English arm "
     "from matching inside the 'provenance' record-of-origin identifier)"),
]

COMPLIANCE_MARKERS: list[tuple[str, str]] = [
    ("no_reliable",
     r"no reliable inference|无法推断|不可作可靠推断|lacks calibrated"),
    ("not_statistical",
     r"非统计显著性证明|非市场规律|not a statistical"),
    ("resource_gate", r"研发资源分配门槛"),
    ("forbidden_to_publish",
     r"不可作为|不得作为|不得支持|not a verification|unsupported"),
    ("negative_verb",
     r"未生成|未运行|未触发|未启用|did not (?:run|generate|trigger|enable)"
     r"|not (?:run|generated|triggered|enabled)"),
    ("descriptive",
     r"描述性|descriptive|开发验证|development validation"),
    ("deferred", r"deferred|推迟|known[_ ]issue"),
    ("fail_closed",
     r"qualified\s*=\s*False|not_qualified|EVIDENCE_NOT_REGISTERED"
     r"|fail[-\s]closed"),
    ("measurement_inc", r"measurement_incomplete|measurement incomplete"),
    ("mechanical", r"机械筛选|mechanical"),
    ("no_promotion",
     r"no promotion|不晋级|未晋级|cannot support promotion|pre-wp17"),
]

#: p57's own artifacts inside the wp18 tree — this audit's outputs and
#: the closure package's recon/verification/report notes, not WP18
#: products (the audit targets the frozen WP18 corpus).
EXCLUDED_NAMES = {
    "claim_audit.json",
    "closure_verifications.json",
    "audit_findings_p57.md",
    "_closure_recon.md",
    "P57_CLOSURE_REPORT.md",
    "WP18_CLOSURE.md",
}

#: Append-only errata dispositions for suspected_inferential hits:
#: "(relative_path, line, pattern_id)" -> errata entry id.
DISPOSITIONS: dict[str, str] = {}

_COMPILED = [
    (pid, re.compile(rx, re.IGNORECASE), tier, basis)
    for pid, rx, tier, basis in INFERENTIAL_PATTERNS
]
_MARKERS = [(mid, re.compile(rx, re.IGNORECASE)) for mid, rx in COMPLIANCE_MARKERS]
_KEY_FINDER = re.compile(r'"([^"]+)"\s*:')


def _key_spans(line: str) -> list[tuple[int, int]]:
    return [m.span(1) for m in _KEY_FINDER.finditer(line)]


def scan_text(text: str, file_label: str) -> list[dict]:
    """Scan one product's text; return one record per pattern hit."""

    hits: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        keys = _key_spans(line)
        taken: list[tuple[int, int]] = []
        for pid, rx, tier, basis in _COMPILED:
            for m in rx.finditer(line):
                span = m.span()
                if any(s <= span[0] and span[1] <= e for s, e in taken):
                    continue  # an earlier (higher-priority) pattern owns it
                taken.append(span)
                window_start = max(0, span[0] - WINDOW)
                window_end = min(len(line), span[1] + WINDOW)
                context = line[window_start:window_end]
                marker = next(
                    (mid for mid, mrx in _MARKERS if mrx.search(context)), None
                )
                is_key = any(
                    s <= span[0] and span[1] <= e for s, e in keys
                )
                if is_key:
                    classification, reason = (
                        "descriptive_compliant", "json_field_name"
                    )
                elif marker is not None:
                    classification, reason = (
                        "descriptive_compliant", f"marker:{marker}"
                    )
                elif tier == "strong":
                    classification, reason = (
                        "suspected_inferential", "unmarked_strong_pattern"
                    )
                else:
                    classification, reason = (
                        "needs_annotation", "unmarked_ambient_pattern"
                    )
                hits.append(
                    {
                        "file": file_label,
                        "line": lineno,
                        "pattern_id": pid,
                        "tier": tier,
                        "matched_text": m.group(0),
                        "classification": classification,
                        "reason": reason,
                        "context": context,
                        "basis": basis,
                    }
                )
    hits.sort(key=lambda h: (h["file"], h["line"], h["pattern_id"]))
    return hits


def _iter_product_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".md", ".json", ".jsonl"):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        yield path


def run_audit(root: Path = WP18) -> dict:
    """Scan every wp18 MD/JSON product; classify each hit; attach the
    append-only errata dispositions for suspected inferential wording."""

    files = []
    hits: list[dict] = []
    for path in _iter_product_files(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        files.append(rel)
        hits.extend(scan_text(path.read_text(encoding="utf-8"), rel))
    by_classification = {
        label: sum(1 for h in hits if h["classification"] == label)
        for label in (
            "descriptive_compliant", "needs_annotation", "suspected_inferential"
        )
    }
    suspected = [h for h in hits if h["classification"] == "suspected_inferential"]
    for hit in suspected:
        key = f"({hit['file']}, {hit['line']}, {hit['pattern_id']})"
        hit["errata_disposition"] = DISPOSITIONS.get(key)
    unhandled = [
        h for h in suspected if not h.get("errata_disposition")
    ]
    return {
        "audit": "p57_wp18_claim_audit",
        "basis": (
            "FROZEN_17 wording contract (reports/wp17/FROZEN_17.md §3.3/§3.4 "
            "templates, §4.1/§4.3 banned wording) + the WP18 report footer "
            "(descriptive development validation; 研发资源分配门槛; "
            "非统计显著性证明也非市场规律; no nominal-5% significance or "
            "out-of-sample inference publishing).  Pure text scan: zero "
            "backtests, zero measurement rows."
        ),
        "pattern_word_list": [
            {"id": pid, "regex": rx, "tier": tier, "basis": basis}
            for pid, rx, tier, basis in INFERENTIAL_PATTERNS
        ],
        "compliance_markers": [
            {"id": mid, "regex": rx} for mid, rx in COMPLIANCE_MARKERS
        ],
        "excluded": sorted(EXCLUDED_NAMES),
        "files_scanned": files,
        "n_files": len(files),
        "hits": hits,
        "summary": {
            "files": len(files),
            "hits": len(hits),
            "by_classification": by_classification,
            "suspected_inferential_total": len(suspected),
            "suspected_inferential_unhandled": len(unhandled),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=WP18, help="wp18 products root"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="audit output JSON"
    )
    args = parser.parse_args()
    result = run_audit(args.root)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"p57 claim audit: {result['summary']['files']} files, "
        f"{result['summary']['hits']} hits, "
        f"suspected_inferential_unhandled="
        f"{result['summary']['suspected_inferential_unhandled']} -> "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
