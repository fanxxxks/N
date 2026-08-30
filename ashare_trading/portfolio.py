"""Simulated A-share portfolio state.

Persistence is a typed-artifact boundary (P7-C,
docs/p7_artifact_schema_contract.md §5): saves are stamped with
``artifact_schema_version`` and validated fail-closed; loads run the
schema matrix — a corrupt or unknown-version state file raises
:class:`ashare_model.artifact_schemas.ArtifactSchemaError` and is *never*
overwritten by a fresh state (the pre-contract ``except Exception ->
reset()`` path could destroy the only copy of an account).  Legacy
version-less state files load through the pre-contract tolerant path and
migrate on the next normal save.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ashare_data.io_utils import atomic_write_json
from ashare_model.artifact_schemas import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaError,
    PaperStateArtifact,
    apply_schema_matrix,
)


@dataclass
class PositionState:
    ts_code: str
    name: str
    quantity: float
    available_quantity: float
    avg_cost: float
    last_price: float
    last_date: str


class SimulationPortfolio:
    def __init__(
        self,
        initial_capital: float,
        state_path: str | Path,
    ):
        self.initial_capital = float(initial_capital)
        self.state_path = Path(state_path)
        self.cash = self.initial_capital
        self.positions: dict[str, PositionState] = {}
        self.equity_history: list[dict[str, float | str]] = []
        self.trade_count = 0
        # Last fully processed execution date (YYYYMMDD). This is the resume
        # watermark: a day is only recorded here after its orders/trades were
        # written and its equity snapshot saved, so replay can never overlap.
        self.last_exec_date: str | None = None
        self.load()

    @property
    def has_history(self) -> bool:
        """True when the state carries anything a replay could corrupt."""
        return bool(
            self.last_exec_date
            or self.equity_history
            or self.positions
            or self.trade_count
        )

    def reset(self) -> None:
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_history = []
        self.trade_count = 0
        self.last_exec_date = None
        self.save()

    def load(self) -> None:
        if not self.state_path.exists():
            return
        # Fail-closed (P7-C §5): a corrupt, non-dict or unknown/future
        # version state file raises and is never overwritten by reset();
        # resume stops for manual disposition with the backup intact.
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArtifactSchemaError(
                f"paper state {self.state_path} is corrupt or unreadable; "
                f"refusing to resume or overwrite it ({exc})"
            ) from exc
        if not isinstance(payload, dict):
            raise ArtifactSchemaError(
                f"paper state {self.state_path} is not a JSON object; "
                "refusing to resume or overwrite it"
            )
        # Version matrix: unknown/future hard-rejects, current validates,
        # legacy (no version key) flows through the pre-contract path.
        apply_schema_matrix(
            payload, artifact="paper state", model=PaperStateArtifact
        )
        try:
            self.cash = float(payload.get("cash", self.initial_capital))
            self.trade_count = int(payload.get("trade_count", 0))
            self.positions = {
                k: PositionState(**v) for k, v in payload.get("positions", {}).items()
            }
            self.equity_history = payload.get("equity_history", [])
            last = payload.get("last_exec_date")
            if not last and self.equity_history:
                # Legacy state files predate last_exec_date; the tail of the
                # equity history is the best available resume watermark.
                last = self.equity_history[-1].get("trade_date")
            self.last_exec_date = last
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ArtifactSchemaError(
                f"paper state {self.state_path} has malformed fields; "
                f"refusing to resume or overwrite it ({exc})"
            ) from exc

    def save(self) -> None:
        payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "trade_count": self.trade_count,
            "last_exec_date": self.last_exec_date,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "equity_history": self.equity_history,
        }
        # Fail-closed (P7-C §2.1): a state that does not satisfy the
        # schema raises ValidationError here and never reaches disk.
        PaperStateArtifact.model_validate(payload)
        # Atomic write: a crash mid-save can never leave a truncated state.
        atomic_write_json(self.state_path, payload)

    def add_buy(self, ts_code: str, name: str, quantity: float, price: float, date: str) -> None:
        cost = quantity * price
        self.cash -= cost
        pos = self.positions.get(ts_code)
        if pos is None:
            self.positions[ts_code] = PositionState(
                ts_code=ts_code,
                name=name,
                quantity=quantity,
                available_quantity=0,
                avg_cost=price,
                last_price=price,
                last_date=date,
            )
        else:
            old_qty = pos.quantity
            total_cost = old_qty * pos.avg_cost + cost
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity else 0.0
            pos.last_price = price
            pos.last_date = date
        self.trade_count += 1

    def add_sell(self, ts_code: str, quantity: float, price: float, date: str) -> None:
        pos = self.positions.get(ts_code)
        if pos is None:
            return
        quantity = min(quantity, pos.available_quantity)
        if quantity <= 0:
            return
        self.cash += quantity * price
        pos.quantity -= quantity
        pos.available_quantity -= quantity
        pos.last_price = price
        pos.last_date = date
        if pos.quantity <= 0:
            self.positions.pop(ts_code, None)
        self.trade_count += 1

    def mark_new_day(self) -> None:
        for pos in self.positions.values():
            pos.available_quantity = pos.quantity

    def market_value(self, prices: dict[str, float]) -> float:
        value = 0.0
        for ts_code, pos in self.positions.items():
            price = prices.get(ts_code, pos.last_price)
            pos.last_price = price
            value += pos.quantity * price
        return value

    def record_equity(self, date: str, prices: dict[str, float]) -> float:
        equity = self.cash + self.market_value(prices)
        self.equity_history.append({"trade_date": date, "equity": equity})
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-10000:]
        return equity
