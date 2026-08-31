"""Simulated A-share portfolio state.

Schema-v2 saves are lifecycle-bound RunStore artifacts. A portfolio must
be explicitly bound to an open run, candidate and account before saving;
the historical state path is only an atomic convenience mirror. Corrupt
or unknown files fail closed. Legacy v0/v1 state remains readable for
audit, but is read-only and can never be implicitly migrated or resumed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ashare_model.artifact_schemas import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaError,
    PaperStateArtifact,
    apply_schema_matrix,
)
from ashare_model.artifact_writer import write_boundary_artifact
from ashare_model.run_store import RunHandle


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
        self._handle: RunHandle | None = None
        self._bound_candidate_id: str | None = None
        self._bound_account_id: str | None = None
        self._loaded_lineage: dict[str, str] | None = None
        self._legacy_read_only = False
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

    @property
    def loaded_lineage(self) -> dict[str, str] | None:
        """Identity loaded from the current state mirror, if any."""

        return dict(self._loaded_lineage) if self._loaded_lineage else None

    @property
    def legacy_read_only(self) -> bool:
        return self._legacy_read_only

    def bind_lineage(
        self,
        handle: RunHandle,
        *,
        candidate_id: str,
        account_id: str,
    ) -> None:
        """Bind saves to one new follower run and validate resume lineage."""

        if self._legacy_read_only:
            raise ArtifactSchemaError(
                "legacy v0/v1 paper state is audit-only and cannot be "
                "bound, resumed or saved as schema v2"
            )
        if not isinstance(handle, RunHandle):
            raise ArtifactSchemaError(
                "paper state binding requires an open RunHandle (RunStore)"
            )
        if self._handle is not None:
            raise ArtifactSchemaError(
                "paper state is already bound; detach the prior run handle first"
            )
        expected = {
            "spec_id": handle.spec.spec_id,
            "candidate_id": candidate_id,
            "account_id": account_id,
        }
        if self._loaded_lineage is not None:
            conflicts = {
                key: {
                    "state": self._loaded_lineage.get(key),
                    "strategy": value,
                }
                for key, value in expected.items()
                if self._loaded_lineage.get(key) != value
            }
            if conflicts:
                raise ArtifactSchemaError(
                    "paper state/strategy lineage mismatch; refusing resume: "
                    f"{conflicts}"
                )
        # Validate identity formats and the in-memory state before the day
        # loop starts. The formal writer repeats this check before disk I/O.
        PaperStateArtifact.validate_payload(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "spec_id": handle.spec.spec_id,
                "run_id": handle.run_id,
                "candidate_id": candidate_id,
                "account_id": account_id,
                **self._state_payload(),
            }
        )
        self._handle = handle
        self._bound_candidate_id = candidate_id
        self._bound_account_id = account_id

    def detach_handle(self) -> None:
        """Detach without closing; the run owner closes the handle."""

        self._handle = None
        self._bound_candidate_id = None
        self._bound_account_id = None

    def reset(self, *, allow_legacy: bool = False) -> None:
        if self._legacy_read_only and not allow_legacy:
            raise ArtifactSchemaError(
                "legacy v0/v1 paper state is audit-only; archive/retire it "
                "before an explicit reset"
            )
        if self._handle is not None:
            raise ArtifactSchemaError(
                "cannot reset paper state while a RunStore handle is bound"
            )
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ArtifactSchemaError(
                f"could not remove paper state mirror {self.state_path}: {exc}"
            ) from exc
        self.cash = self.initial_capital
        self.positions = {}
        self.equity_history = []
        self.trade_count = 0
        self.last_exec_date = None
        self._loaded_lineage = None
        self._legacy_read_only = False

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
        verdict = apply_schema_matrix(
            payload, artifact="paper state", model=PaperStateArtifact
        )
        self._legacy_read_only = verdict == "legacy"
        if verdict == "current":
            self._loaded_lineage = {
                key: str(payload[key])
                for key in ("spec_id", "run_id", "candidate_id", "account_id")
            }
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

    def _state_payload(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "trade_count": self.trade_count,
            "last_exec_date": self.last_exec_date,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "equity_history": self.equity_history,
        }

    def save(self) -> None:
        if self._legacy_read_only:
            raise ArtifactSchemaError(
                "legacy v0/v1 paper state is audit-only; refusing implicit "
                "schema-v2 migration or save"
            )
        if (
            self._handle is None
            or self._bound_candidate_id is None
            or self._bound_account_id is None
        ):
            raise ArtifactSchemaError(
                "paper state is not bound to a RunStore run, candidate and account"
            )
        write_boundary_artifact(
            self._handle,
            artifact_type="paper_state",
            model_cls=PaperStateArtifact,
            payload=self._state_payload(),
            candidate_id=self._bound_candidate_id,
            account_id=self._bound_account_id,
            convenience_path=self.state_path,
        )
        self._loaded_lineage = {
            "spec_id": self._handle.spec.spec_id,
            "run_id": self._handle.run_id,
            "candidate_id": self._bound_candidate_id,
            "account_id": self._bound_account_id,
        }

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
