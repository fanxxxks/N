"""Registry docs drift guard (P7 D3, plan §6).

``docs/feature_registry.md`` is **generated** from the single registries
(feature metadata, research domains, data tiers, operator registry) by
``scripts/generate_registry_docs.py`` — it must never be hand-edited.
This guard fails when the committed document no longer matches what the
generators produce: fix the drift by re-running the generator (and
committing code + doc together), never by editing the markdown.

The hand-maintained feature/operator lists in PROJECT_ONBOARDING §5 are
retired in favor of this generated reference; a name list reappearing
there would recreate a second authority, so the guard also asserts the
onboarding sections point at the generated file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "feature_registry.md"
SCRIPT_PATH = ROOT / "scripts" / "generate_registry_docs.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_registry_docs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_doc_matches_registries():
    module = _load_generator()
    assert DOC_PATH.exists(), (
        "docs/feature_registry.md is missing; run "
        "`python scripts/generate_registry_docs.py` and commit the result"
    )
    expected = module.build_document()
    actual = DOC_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/feature_registry.md is stale; regenerate with "
        "`python scripts/generate_registry_docs.py` and commit code+doc together"
    )


def test_onboarding_links_the_generated_reference():
    text = (ROOT / "docs" / "PROJECT_ONBOARDING.md").read_text(encoding="utf-8")
    assert "feature_registry.md" in text, (
        "PROJECT_ONBOARDING must link the generated registry reference"
    )
    # The retired hand-maintained operator table must not come back: the
    # full operator name list may only live in the generated doc.
    assert "TS_RANK5、TS_RANK10、TS_RANK20、TS_RANK60" not in text, (
        "hand-maintained operator name lists are retired (P7 D3); "
        "link docs/feature_registry.md instead"
    )
