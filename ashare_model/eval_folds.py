"""Walk-forward fold/window resolution for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A1) by reason-to-change: this
module owns how fold configs become concrete date-indexed contracts and how
a fold's test window is sliced from the loader.  It changes when the
fold/window *contract* changes — not when metrics, statistical corrections,
search backends or artifact schemas change.

The module is import-leaf-ward of the facade: it must never import
``ashare_model.evaluation`` (consumers reach these names through the facade
re-exports, preserving the pre-split import surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from ashare_data.config import FoldConfig
from ashare_data.processor import open_to_open_returns
from ashare_portfolio.rebalance import RebalancePolicy

from .data_loader import AshareDataLoader
from .research_domain import UNIFIED_DOMAIN_ID
from .targets import causal_target_returns
from .time_contract import FoldTimeContract


@dataclass(frozen=True)
class Fold:
    """A walk-forward fold resolved against a concrete date axis."""

    contract: FoldTimeContract
    frequency: str = "daily"

    @property
    def policy(self) -> RebalancePolicy:
        return RebalancePolicy(self.frequency, self.contract.horizon)

    @property
    def train_end(self) -> str:
        return self.contract.train_end

    @property
    def test_end(self) -> str:
        return self.contract.test_end

    # Compatibility accessors for consumers of pre-v7 Fold. New code uses
    # the explicit contract fields so anchor, signal and price ends cannot be
    # confused.
    @property
    def train_end_idx(self) -> int:
        return self.contract.train_anchor_end_exclusive

    @property
    def test_end_idx(self) -> int:
        return self.contract.test_price_end


def search_window_id(
    fold: Fold, seed: int, domain_id: str | None = None
) -> str:
    """Deterministic search-window id for one (fold, seed) evaluation.

    A non-unified research domain (P6 §4.3) appends ``domain:<id>`` so
    semantic-cache scores of one domain never mix with unified or
    other-domain scores; the default id stays pre-P6 byte-identical.
    """

    domain = (
        f":domain:{str(domain_id)}"
        if domain_id is not None and str(domain_id) != UNIFIED_DOMAIN_ID
        else ""
    )
    return (
        f"fold:{fold.train_end}:{fold.test_end}:"
        f"frequency:{fold.policy.frequency}:horizon:{fold.policy.horizon}:"
        f"seed:{seed}{domain}"
    )


@dataclass(frozen=True)
class FoldData:
    """Price-context slice plus the contract that declares executable columns."""

    factors: np.ndarray
    raw: dict[str, np.ndarray]
    target: np.ndarray
    realized_ret: np.ndarray
    rebalance_mask: np.ndarray
    dates: list[str]
    universe_mask: np.ndarray
    contract: FoldTimeContract

    @property
    def signal_count(self) -> int:
        return self.contract.test_signal_count

    @property
    def local_signal_range(self) -> range:
        return range(self.signal_count)

    def __iter__(self):
        # Preserve the established four-value unpacking API while exposing
        # the contract to new callers as an explicit attribute.
        yield self.factors
        yield self.raw
        yield self.target
        yield self.dates


def resolve_folds(
    fold_cfgs: list[FoldConfig],
    dates: list[str],
    *,
    frequency: str = "daily",
    horizon: int = 1,
) -> list[Fold]:
    """Resolve fold configs to column indices and check data availability.

    Configured anchors are inclusive. Test data retains the exact
    ``1 + horizon`` price-context columns needed to exit its final executable
    signal, while neither train nor test scoring can observe a price beyond
    its anchor.
    """

    policy = RebalancePolicy(frequency, horizon)
    folds: list[Fold] = []
    for cfg in fold_cfgs:
        contract = FoldTimeContract.resolve(
            dates,
            train_end=cfg.train_end,
            test_end=cfg.test_end,
            horizon=policy.horizon,
        )
        if (
            contract.test_price_end == len(dates)
            and dates[-1].replace("-", "") < cfg.test_end.replace("-", "")
        ):
            logger.warning(
                f"fold {cfg.train_end} -> {cfg.test_end}: test_end is past the "
                f"data range; test window truncated at {dates[-1]}"
            )
        folds.append(Fold(contract, frequency=policy.frequency))
    return folds


def epoch_slice(
    loader: AshareDataLoader,
    fold: Fold,
) -> FoldData:
    """Factor stack, raw OHLCV cache, forward targets and dates of the test
    window.  Factor columns carry their own lookback, so slicing the test
    window loses no history (VM execution must still happen on the full
    tensor and be sliced afterwards).

    The sparse forward target is recomputed from the sliced opens using the
    fold's global schedule slice. Passing the pre-resolved mask prevents a
    5/10-day cadence from restarting at the fold boundary.
    """

    contract = fold.contract
    s0, s1 = contract.test_signal_start, contract.test_price_end
    if loader.universe_mask is None:
        raise ValueError(
            "loader carries no universe mask; production evaluation "
            "requires the PIT eligibility mask"
        )
    factors = loader.factor_tensor[:, :, s0:s1].numpy()
    raw = {k: v[:, s0:s1].numpy() for k, v in loader.raw_data_cache.items()}
    universe_mask = loader.universe_mask[:, s0:s1]
    rebalance_mask = fold.policy.rebalance_mask(loader.dates)[s0:s1]
    target = causal_target_returns(
        raw["open"],
        loader.dates[s0:s1],
        fold.policy,
        rebalance_mask=rebalance_mask,
    )
    target = loader.mask_by_universe(target, start=s0)
    realized_ret = open_to_open_returns(raw["open"])
    return FoldData(
        factors=factors,
        raw=raw,
        target=target,
        realized_ret=realized_ret,
        rebalance_mask=rebalance_mask,
        dates=loader.dates[s0:s1],
        universe_mask=universe_mask,
        contract=contract,
    )
