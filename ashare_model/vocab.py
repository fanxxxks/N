"""A-share factor/operator vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

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


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
