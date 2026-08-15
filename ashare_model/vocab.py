"""A-share factor/operator vocabulary.

The vocabulary is versioned: :attr:`FormulaVocab.feature_version` is a hash
of the feature/operator name lists, recorded in every training artifact.
Saved formulas are resolved against the current vocabulary *by name* (see
:func:`resolve_formula_tokens`), so vocabulary additions never silently
reinterpret token ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from .ops import OPS_CONFIG


FEATURE_NAMES = (
    "RET_1",
    "RET_5",
    "RET_10",
    "RET_20",
    "VOL_20",
    "VOL_60",
    "TURNOVER",
    "TURNOVER_CHG",
    "VOLUME_RATIO",
    "VOLUME_IMPACT",
    "AMPLITUDE",
    "CLOSE_POSITION",
    "MOMENTUM_20",
    "MOMENTUM_60",
    "REVERSAL_5",
    "SKEW_20",
    "KURT_20",
    "PE_TTM",
    "PB",
    "PS_TTM",
    "ROE",
    "ROA",
    "GROSS_MARGIN",
    "NET_MARGIN",
    "REVENUE_YOY",
    "PROFIT_YOY",
    "DEBT_RATIO",
    "MARKET_CAP",
    "DIVIDEND_YIELD",
    "NORTHBOUND_CHG",
    "MARGIN_BALANCE_CHG",
    "LIMIT_UP_EVENT",
    "LIMIT_DOWN_EVENT",
    "INDUSTRY_MOMENTUM",
)

# Feature list of the first released vocabulary generation.  Formulas saved
# without feature metadata (legacy artifacts) resolve against this pinned
# list; once the current vocabulary no longer matches it, legacy formulas
# are rejected instead of being silently misinterpreted.
LEGACY_FEATURE_NAMES = FEATURE_NAMES


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]
    pad_token_id: int = 0

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return 1 + self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return ("PAD",) + self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)

    @property
    def feature_version(self) -> str:
        """Short hash of the feature/operator name lists.

        Stored in training artifacts so a formula can always be mapped back
        to the vocabulary it was sampled against.
        """
        payload = json.dumps(
            {
                "features": list(self.feature_names),
                "operators": list(self.operator_names),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)


def tokens_to_names(tokens: Iterable[int], vocab: FormulaVocab | None = None) -> list[str] | None:
    """Map a postfix token sequence to its token names.

    Returns ``None`` when a token id is out of range for the vocabulary
    (structurally invalid sequence).
    """

    vocab = vocab or FORMULA_VOCAB
    names: list[str] = []
    for raw in tokens:
        token = int(raw)
        if token == vocab.pad_token_id:
            names.append("PAD")
            continue
        if 1 <= token < vocab.size:
            names.append(vocab.token_names[token])
        else:
            return None
    return names


def resolve_formula_tokens(payload, vocab: FormulaVocab | None = None) -> list[int]:
    """Resolve a saved formula payload to token ids in the current vocabulary.

    Payloads written by the trainer record the feature/operator name lists
    they were sampled against (``feature_names`` / ``operator_names`` /
    ``feature_version``), so formulas are remapped **by name** and survive
    vocabulary additions.  Legacy payloads without metadata resolve against
    :data:`LEGACY_FEATURE_NAMES` and are rejected once the current
    vocabulary no longer matches that pinned list.  A bare token list is
    accepted as a legacy payload.
    """

    vocab = vocab or FORMULA_VOCAB
    if isinstance(payload, (list, tuple)):
        payload = {"formula": list(payload)}
    if not isinstance(payload, dict) or "formula" not in payload:
        raise ValueError("formula payload has no 'formula' field")

    src_features = payload.get("feature_names")
    if src_features is None:
        if tuple(vocab.feature_names) != tuple(LEGACY_FEATURE_NAMES):
            raise ValueError(
                "saved formula has no feature metadata and the current "
                "vocabulary no longer matches the legacy feature set; "
                "re-train the formula instead of silently reinterpreting tokens"
            )
        src_features = LEGACY_FEATURE_NAMES
    src_features = tuple(str(name) for name in src_features)

    src_operators = payload.get("operator_names")
    if src_operators is None:
        src_operators = vocab.operator_names
    src_operators = tuple(str(name) for name in src_operators)

    src_vocab = FormulaVocab(feature_names=src_features, operator_names=src_operators)
    names = tokens_to_names(payload["formula"], src_vocab)
    if names is None:
        raise ValueError("formula tokens out of range for the recorded vocabulary")

    resolved: list[int] = []
    for name in names:
        if name == "PAD":
            resolved.append(vocab.pad_token_id)
        elif name in vocab.feature_names:
            resolved.append(1 + vocab.feature_names.index(name))
        elif name in vocab.operator_names:
            resolved.append(vocab.operator_offset + vocab.operator_names.index(name))
        else:
            raise ValueError(
                f"formula references '{name}', which is missing from the "
                "current vocabulary; re-train or restore the matching feature set"
            )
    return resolved
