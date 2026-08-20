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


# First vocabulary generation (v1): the original 34 features.  Kept as a
# separate constant so legacy formulas saved without metadata can always be
# remapped by name, even after later generations grow the vocabulary.
_FEATURE_NAMES_V1 = (
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

# Second vocabulary generation (v2): local price-volume families appended
# after the v1 features so the v1 token ids never shift.
_FEATURE_NAMES_V2 = (
    # Intraday decomposition (open/close vs pre_close; all local).
    "OVERNIGHT_RET",
    "INTRADAY_RET",
    # Liquidity (local; free-float turnover is already covered by TURNOVER).
    "ILLIQ_20",
    "AMOUNT_SHARE",
    # Lottery / anchoring.
    "MAX_20",
    "HIGH_52W",
    # Market-model risk family (rolling CAPM vs the equal-weight market).
    "BETA_60",
    "IVOL_60",
    "RSQ_60",
    # Technical (BIAS is the moving-average distance; MACD keeps DIF/DEA,
    # HIST is expressible as DIF SUB DEA).
    "BIAS_20",
    "RSI_14",
    "ATR_14",
    "MACD_DIF",
    "MACD_DEA",
    # Microstructure (local).
    "SUSPEND_DAYS_60",
    "LIST_AGE",
)

# Third vocabulary generation (v3): factor P0 additions, all local
# computation (no new data interfaces).  Medium-term windows, smoothed
# turnover, limit-move statistics and industry-relative versions of the
# strongest families.
_FEATURE_NAMES_V3 = (
    # Medium-term momentum / reversal (slow, low-turnover signals).
    "RET_120",
    "REVERSAL_60",
    "REVERSAL_120",
    # Smoothed turnover: 5/20-day means and 20-day volatility.
    "TURNOVER_MA5",
    "TURNOVER_MA20",
    "TURNOVER_STD20",
    # Limit-move statistics (event family, all local).
    "LIMIT_STREAK",
    "LIMIT_UP_CNT_20",
    "LIMIT_BREAK",
    # Industry-relative versions of the strongest families (industry
    # membership from the Shenwan snapshot, demeaned per industry).
    "IND_REL_RET_5",
    "IND_REL_RET_20",
    "IND_REL_VOL_20",
    "IND_REL_TURNOVER",
)

# Retired feature names, resolvable to their surviving equivalents: a saved
# formula that references a retired name keeps its exact semantics, so no
# artifact ever breaks when a duplicate is removed from the live vocabulary.
# RET_20 was identical to MOMENTUM_20 (same computation, 0.99 correlation).
FEATURE_ALIASES = {"RET_20": "MOMENTUM_20"}

FEATURE_NAMES = tuple(
    name
    for name in _FEATURE_NAMES_V1 + _FEATURE_NAMES_V2 + _FEATURE_NAMES_V3
    if name not in FEATURE_ALIASES
)

# Pins of the first released vocabulary generation.  Formulas saved without
# feature metadata (legacy artifacts) resolve against these lists; the
# remapping is by name, so it stays correct after vocabulary additions.
LEGACY_FEATURE_NAMES = _FEATURE_NAMES_V1

# Operator list of the first released vocabulary generation, pinned for the
# same reason: legacy formulas carry operator token ids in the pre-growth
# layout and are remapped by name against this list.
LEGACY_OPERATOR_NAMES = (
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "NEG",
    "ABS",
    "SIGN",
    "GATE",
    "JUMP",
    "DECAY",
    "DELAY1",
    "MAX3",
    "DELTA5",
    "MA20",
    "STD20",
    "TS_RANK20",
)


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
    the pinned first-generation lists (:data:`LEGACY_FEATURE_NAMES` /
    :data:`LEGACY_OPERATOR_NAMES`); the by-name remapping keeps them valid
    after vocabulary growth.  A bare token list is accepted as a legacy
    payload.
    """

    vocab = vocab or FORMULA_VOCAB
    if isinstance(payload, (list, tuple)):
        payload = {"formula": list(payload)}
    if not isinstance(payload, dict) or "formula" not in payload:
        raise ValueError("formula payload has no 'formula' field")

    src_features = payload.get("feature_names")
    if src_features is None:
        src_features = LEGACY_FEATURE_NAMES
    src_features = tuple(str(name) for name in src_features)

    src_operators = payload.get("operator_names")
    if src_operators is None:
        src_operators = LEGACY_OPERATOR_NAMES
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
        elif name in FEATURE_ALIASES and FEATURE_ALIASES[name] in vocab.feature_names:
            resolved.append(1 + vocab.feature_names.index(FEATURE_ALIASES[name]))
        elif name in vocab.operator_names:
            resolved.append(vocab.operator_offset + vocab.operator_names.index(name))
        else:
            raise ValueError(
                f"formula references '{name}', which is missing from the "
                "current vocabulary; re-train or restore the matching feature set"
            )
    return resolved
