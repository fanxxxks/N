"""A-share factor/operator vocabulary.

The vocabulary is versioned: :attr:`FormulaVocab.feature_version` is a hash
of the feature/operator name lists *and* the grammar generation, recorded
in every training artifact.  Saved formulas are resolved against the
current vocabulary *by name* (see :func:`resolve_formula_tokens`), so
vocabulary additions never silently reinterpret token ids.

Grammar generations:

* v1 (open_slots era): ``PAD`` only, no ``EOS`` token; the sampling mask
  mixed an ``open_slots`` counter with the stack (removed).
* v2 (current): an independent ``EOS`` token terminates every formula;
  ``PAD`` is legal only after ``EOS``; postfix legality is stack-only
  (see :mod:`ashare_model.ir`).  ``EOS`` sits at the end of the token
  space so pre-v2 feature/operator ids never shift.
* v3 (P7-E): sampling legality gains the semantic-type dimension — the
  action mask is type-aware (see
  ``docs/p7_semantic_types_contract.md``).
* v4 (P9): the four orthogonal P9 families join the vocabulary and the
  deprecated features leave the sampling space
  (docs/p9_factor_family_contract.md).  Execution, canonicalization and
  by-name remapping of saved formulas are unchanged.
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

# Third vocabulary generation (v3) ends at the P7-E state; v4 (P9,
# docs/p9_factor_family_contract.md §5) appends the four approved orthogonal
# families: industry-residualized momentum, liquidity shock / volume
# shrinkage / price-volume divergence, limit-event conditioning, and
# per-stock crowding.  Names are appended so every pre-v4 token id is
# stable (the same discipline as v2/v3).
_FEATURE_NAMES_V4 = (
    # Family ①: industry-residualized medium-horizon momentum/reversal.
    "IND_REL_RET_60",
    "IND_REL_RET_120",
    # Family ②: liquidity shock, volume shrinkage, price-volume divergence.
    "LIQ_SHOCK_20",
    "VOLUME_SHRINK_5_20",
    "PV_DIV_20",
    # Family ③: limit-event conditioning (sparse-safe standardization).
    "LIMIT_UP_CNT_5",
    "LIMIT_DOWN_STREAK",
    "LIMIT_BREAK_5",
    # Family ④: per-stock crowding (breadth is out of the vocabulary).
    "CROWD_TURNOVER_60",
    "CROWD_AMOUNT_60",
    "MARGIN_CROWD_60",
)

# v5 (P13, docs/p13_fundamental_fields_contract.md §5.3) registers the four
# family-⑤ slow-fundamental features (cash-flow quality, accruals, asset
# growth, earnings acceleration) once their PIT data fields are registered.
# t46 B-fix: they sit at the END of the WHOLE token table -- AFTER the
# operators and EOS -- so every grammar-5-era token id (features 1-73,
# operators 74-112, EOS 113) keeps its exact meaning; appending them to the
# feature block instead shifted the operator/EOS ids and broke legacy
# formula decoding (t28 integration, seed-7 live regression).  The family
# joins and leaves promotion as one atomic unit (§8 族级裁决).
_FEATURE_NAMES_V5 = (
    "CASHFLOW_QUALITY",
    "ACCRUALS",
    "ASSET_GROWTH",
    "EARNINGS_ACCEL",
)

# Head features occupy the contiguous pre-operator feature block (token ids
# 1-73, unchanged since grammar v5); the family-⑤ tail rides after EOS.
FEATURE_NAMES_HEAD = tuple(
    name
    for name in _FEATURE_NAMES_V1 + _FEATURE_NAMES_V2 + _FEATURE_NAMES_V3 + _FEATURE_NAMES_V4
    if name not in FEATURE_ALIASES
)

FEATURE_NAMES = FEATURE_NAMES_HEAD + _FEATURE_NAMES_V5

# P9 §4: approved deprecations (docs/p9_factor_family_contract.md §4.1,
# APPROVED 2026-09-01).  Deprecated names keep their token ids and their
# factor computations (legacy formulas resolve by name forever); they only
# leave the sampling space (grammar v4 action mask) and promotion.
# P9 §7 adjudication (measurement log 2026-09-01): LIMIT_STREAK's
# conditional deprecation TRIGGERED (|corr| with LIMIT_UP_EVENT = 0.980),
# and two second-pass consolidations joined the set.
DEPRECATION_REASONS: dict[str, str] = {
    "NORTHBOUND_CHG": "daily feed discontinued (2024-08); stays neutral",
    "RET_5": "P9 §4.1: exact mirror of REVERSAL_5 (|corr|=1.000); NEG(REVERSAL_5) is equivalent",
    "RET_10": "P9 §4.1: 0.902 correlated with BIAS_20; covered by RET_1 + BIAS_20",
    "RET_120": "P9 §4.1: exact mirror of REVERSAL_120 (|corr|=1.000); weaker standalone ICIR",
    "MOMENTUM_60": "P9 §4.1: exact mirror of REVERSAL_60 (|corr|=1.000)",
    "VOLUME_RATIO": "P9 §4.1: 0.977 correlated with VOLUME_IMPACT (log form kept for robustness)",
    "TURNOVER_MA5": "P9 §4.1: 0.926/0.928 correlated with TURNOVER and TURNOVER_MA20",
    "MACD_DEA": "P9 §4.1: 0.926 correlated with MACD_DIF; no independent incremental OOS",
    "VOL_60": "P9 §4.1: 0.907 correlated with IVOL_60 (the stronger incremental contributor)",
    "LIMIT_STREAK": "P9 §7.3 conditional deprecation TRIGGERED post-fix: |corr| with LIMIT_UP_EVENT = 0.980 >= 0.9",
    "LIQ_SHOCK_20": "P9 §7.1 second-pass: 0.968 correlated with VOLUME_IMPACT; marginal family-gate margin (dIC +0.0054)",
    "CROWD_TURNOVER_60": "P9 §7.1 second-pass: 0.971 correlated with CROWD_AMOUNT_60 (the stronger family member)",
}
DEPRECATED_FEATURE_NAMES = frozenset(DEPRECATION_REASONS)

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


# Grammar generation of the current vocabulary.  v1 was the open_slots-era
# layout (PAD only, no EOS); v2 adds the independent EOS token and the
# stack-only postfix grammar; v3 (P7-E) makes the action mask type-aware;
# v4 (P9) appends the four orthogonal P9 families and removes the
# deprecated features from the sampling space (they keep their token ids
# and computations); v5 (P9 adjudication) adds the conditionally-deprecated
# LIMIT_STREAK and the two second-pass consolidations to that set;
# v6 (P13 §5.3/§6.1) appends the four family-⑤ slow-fundamental features
# (append-only, no existing token id moves).
# Bumping this constant changes
# ``feature_version`` and therefore invalidates the token layout recorded
# in older training artifacts (which resolve by name, so they still load).
GRAMMAR_VERSION = 6


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]
    pad_token_id: int = 0
    # Whether the grammar has an independent EOS token.  Legacy source
    # vocabularies (pre-v2 artifacts being remapped by name) are built
    # without one; the live vocabulary always has it.
    has_eos: bool = True
    # P9 §4.2: feature names that left the sampling space but keep their
    # token ids and computations.  Only the live production vocabulary
    # carries these; toy/legacy vocabularies default to an empty set.
    deprecated_names: frozenset[str] = frozenset()
    # P13 t46 B-fix: features riding at the END of the whole token table
    # (AFTER the EOS token, ids eos+1..).  They keep the pre-operator
    # feature block and every legacy token id stable; only grammar-6+
    # artifacts may reference them.
    tail_feature_names: tuple[str, ...] = ()

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def feature_offset(self) -> int:
        """Id of the first feature token (immediately after PAD)."""
        return 1

    @property
    def operator_offset(self) -> int:
        return self.feature_offset + self.feature_count

    @property
    def eos_token_id(self) -> int | None:
        """Id of the EOS token; ``None`` for legacy source vocabularies.

        EOS sits directly after the operator block (``feature_offset +
        feature_count + operator_count``), so pre-EOS feature/operator ids
        never shift even when tail features ride behind EOS (t46 B-fix).
        With an empty tail this equals ``size - 1`` -- the historical
        placement every pre-tail vocabulary relies on.
        """
        if not self.has_eos:
            return None
        return self.feature_offset + len(self.feature_names) + len(self.operator_names)

    @property
    def token_names(self) -> tuple[str, ...]:
        names = ("PAD",) + self.feature_names + self.operator_names
        if self.has_eos:
            names = names + ("EOS",)
        return names + self.tail_feature_names

    @property
    def size(self) -> int:
        return len(self.token_names)

    @property
    def feature_version(self) -> str:
        """Short hash of the feature/operator name lists and the grammar.

        Stored in training artifacts so a formula can always be mapped back
        to the vocabulary and grammar it was sampled against.  A grammar
        bump (e.g. v1 open_slots layout -> v2 EOS grammar) changes this
        hash even when the name lists are unchanged.
        """
        payload = json.dumps(
            {
                "grammar": GRAMMAR_VERSION if self.has_eos else 1,
                "features": list(self.feature_names),
                "operators": list(self.operator_names),
                "tail_features": list(self.tail_feature_names),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES_HEAD,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
    deprecated_names=DEPRECATED_FEATURE_NAMES,
    tail_feature_names=_FEATURE_NAMES_V5,
)

# t46 Plan B (accepted): the frozen grammar-5 layout as a decode-reference
# vocabulary.  Under the B layout the grammar-5 and grammar-6 token tables
# share ids 0-113 (the family-⑤ tail lives at 114-117), so this layout
# interprets both generations' bare token lists correctly and the by-name
# remap is an identity (zero remap cost).  FEATURE_NAMES_HEAD is the
# frozen grammar-5 feature block (append-only by construction).
GRAMMAR_V5_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES_HEAD,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
GRAMMAR_V5_TOKEN_NAMES = GRAMMAR_V5_VOCAB.token_names


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
    ``feature_version`` / ``grammar_version``), so formulas are remapped
    **by name** and survive vocabulary additions.  Legacy payloads without
    metadata resolve against the pinned first-generation lists
    (:data:`LEGACY_FEATURE_NAMES` / :data:`LEGACY_OPERATOR_NAMES`); the
    by-name remapping keeps them valid after vocabulary growth, and a bare
    legacy factor resolves to its :class:`~ashare_model.ir.Feature` token
    (the AST migration happens when the bytecode is decoded).  A bare token
    list is accepted as a legacy payload.

    This function only remaps ids by name; structural validity is the
    grammar's job (:func:`ashare_model.ir.decode`), so structurally invalid
    legacy artifacts keep their historical "resolves, then fails in the VM"
    behavior instead of a new load-time break.
    """

    vocab = vocab or FORMULA_VOCAB
    if isinstance(payload, (list, tuple)):
        payload = {"formula": list(payload)}
    if not isinstance(payload, dict) or "formula" not in payload:
        raise ValueError("formula payload has no 'formula' field")

    # Fail-closed grammar dispatch (t46 captain pre-ruling, AGENTS §4.3):
    # every formal artifact since the P7 schema records
    # ``grammar_version``; a missing or unrecognized generation is
    # rejected from remap decoding instead of being silently decoded
    # against the latest table (that silent decode IS the t28 incident
    # mechanism).  Known generations: 5 and 6.  Under the accepted B
    # layout every grammar-5/6 token id ≤ 113 is bit-stable, so both
    # decode correctly against the live vocabulary with zero remap cost;
    # display layers mark rejected payloads legacy/unavailable (pre-P7
    # artifacts are read-only rejected per §4.6 and need an explicit
    # auditable migration).
    src_grammar_raw = payload.get("grammar_version")
    if src_grammar_raw is None:
        raise ValueError(
            "formula payload has no grammar_version; pre-P7 artifacts are "
            "fail-closed rejected from remap decoding (AGENTS §4.3/§4.6)"
        )
    src_grammar = int(src_grammar_raw)
    if src_grammar not in (5, 6):
        raise ValueError(
            f"unsupported grammar generation {src_grammar} for remap "
            "decoding (known generations: 5, 6)"
        )
    src_has_eos = src_grammar >= 2
    src_features = payload.get("feature_names")
    src_operators = payload.get("operator_names")
    if src_features is None:
        # Bare payloads declare only the generation: under the accepted B
        # layout the grammar-5 and grammar-6 token tables share ids 0-113
        # (the family-⑤ tail lives at 114-117), so the grammar-5 boundary
        # (73 features, operators at 74) interprets both correctly.
        src_features = GRAMMAR_V5_VOCAB.feature_names
        if src_operators is None:
            src_operators = GRAMMAR_V5_VOCAB.operator_names
    src_features = tuple(str(name) for name in src_features)

    src_operators = payload.get("operator_names")
    if src_operators is None:
        src_operators = GRAMMAR_V5_VOCAB.operator_names
    src_operators = tuple(str(name) for name in src_operators)
    # P13 t46 B-fix: grammar-6 artifacts may record the tail block they
    # were sampled against; the key is additive and absent in every
    # pre-tail payload.
    src_tail = tuple(str(name) for name in payload.get("tail_feature_names", ()))

    src_vocab = FormulaVocab(
        feature_names=src_features,
        operator_names=src_operators,
        has_eos=src_has_eos,
        tail_feature_names=src_tail,
    )
    names = tokens_to_names(payload["formula"], src_vocab)
    if names is None:
        raise ValueError("formula tokens out of range for the recorded vocabulary")

    resolved: list[int] = []
    for name in names:
        if name == "PAD":
            resolved.append(vocab.pad_token_id)
        elif name == "EOS":
            if vocab.eos_token_id is None:
                raise ValueError("formula references EOS against a legacy vocabulary")
            resolved.append(vocab.eos_token_id)
        elif name in vocab.feature_names:
            resolved.append(vocab.feature_offset + vocab.feature_names.index(name))
        elif name in FEATURE_ALIASES and FEATURE_ALIASES[name] in vocab.feature_names:
            resolved.append(
                vocab.feature_offset + vocab.feature_names.index(FEATURE_ALIASES[name])
            )
        elif name in vocab.tail_feature_names:
            # P13 t46 B-fix: tail features ride after EOS; ids are stable
            # eos+1.. and never collide with any pre-tail token id.
            if vocab.eos_token_id is None:
                raise ValueError("formula references a tail feature against a legacy vocabulary")
            resolved.append(
                vocab.eos_token_id + 1 + vocab.tail_feature_names.index(name)
            )
        elif name in vocab.operator_names:
            resolved.append(vocab.operator_offset + vocab.operator_names.index(name))
        else:
            raise ValueError(
                f"formula references '{name}', which is missing from the "
                "current vocabulary; re-train or restore the matching feature set"
            )
    return resolved
