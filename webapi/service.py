"""Data access helpers for the AlphaGPT web API.

All reads are defensive: the dashboard must stay alive even while
artifacts are missing, the DuckDB file is locked by a running sync, or
a JSON payload is temporarily truncated by an atomic-write rename.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ashare_data.config import (
    RUNTIME_OVERRIDES_FILENAME,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_sim_config,
    load_config,
)
from ashare_execution import execution_config_mismatches, validate_execution_config
from ashare_data.db import AshareDB
from ashare_data.io_utils import read_json_safe
from ashare_data.universe import UniverseContractError, require_production_universe

ROOT = Path(__file__).resolve().parents[1]

_LOG_DIRS = (ROOT / "logs", ROOT / "data")
_MAX_LOG_READ_BYTES = 16 * 1024 * 1024

_stock_names: dict[str, str] | None = None
_stock_names_mtime: float | None = None


def _load_configs() -> tuple:
    raw = load_config(None, project_root=ROOT)
    data_config = make_data_config(raw, ROOT)
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    sim_config = make_sim_config(raw, ROOT)
    return data_config, model_config, backtest_config, sim_config


def _get_configs() -> tuple:
    try:
        return _load_configs()
    except Exception:  # noqa: BLE001 - dashboard must survive bad config.
        return None, None, None, None


def _read_json(path: Path) -> dict | list | None:
    return read_json_safe(path)


def load_stock_names() -> dict[str, str]:
    """ts_code -> stock name from the local DuckDB.

    Cached with mtime invalidation: a fresh sync (or a renamed stock)
    shows up on the next call without a permanent stale cache.
    """

    global _stock_names, _stock_names_mtime
    data_config, *_ = _get_configs()
    if data_config is None:
        return {}
    db_path = data_config.duckdb_path
    try:
        mtime = db_path.stat().st_mtime if db_path.exists() else None
    except OSError:
        mtime = None
    if _stock_names is not None and mtime == _stock_names_mtime:
        return _stock_names
    try:
        with AshareDB(db_path, read_only=True) as db:
            df = db.query(f"SELECT ts_code, name FROM {data_config.stocks_table}")
        _stock_names = {
            str(row.get("ts_code")): str(row.get("name") or row.get("ts_code"))
            for row in df.to_dict("records")
        }
        _stock_names_mtime = mtime
    except Exception:  # noqa: BLE001
        return {}
    return _stock_names


def get_backtest() -> dict:
    """Full backtest result minus the (huge) positions history."""

    data_config, *_ = _get_configs()
    if data_config is None:
        return {}
    payload = _read_json(data_config.data_dir / "backtest_result.json") or {}
    if not isinstance(payload, dict):
        return {}
    out = {
        k: v
        for k, v in payload.items()
        if k not in {"positions", "trades"}
    }
    positions = payload.get("positions") or []
    out["positions_count"] = len(positions)
    return out


def get_backtest_positions(offset: int = 0, limit: int = 20) -> dict:
    """One page of daily holdings snapshots, newest first, name-enriched."""

    data_config, *_ = _get_configs()
    if data_config is None:
        return {"items": [], "total": 0}
    payload = _read_json(data_config.data_dir / "backtest_result.json") or {}
    positions = payload.get("positions") or [] if isinstance(payload, dict) else []
    names = load_stock_names()
    offset = max(0, int(offset))
    limit = max(1, min(200, int(limit)))
    page = positions[::-1][offset : offset + limit]
    items = []
    for snap in page:
        ts_codes = snap.get("ts_codes") or []
        weights = snap.get("weights") or []
        rows = [
            {
                "ts_code": code,
                "name": names.get(code, ""),
                "weight": round(float(weights[i]), 6) if i < len(weights) else 0.0,
            }
            for i, code in enumerate(ts_codes)
        ]
        items.append(
            {
                "signal_date": snap.get("signal_date"),
                "entry_date": snap.get("entry_date"),
                "exit_date": snap.get("exit_date"),
                "count": len(rows),
                "rows": rows,
            }
        )
    return {"items": items, "total": len(positions)}


def get_strategy() -> dict:
    data_config, *_ = _get_configs()
    if data_config is None:
        return {}
    payload = _read_json(data_config.data_dir / "best_ashare_strategy.json") or {}
    if not isinstance(payload, dict):
        return {"formula": payload}
    # P0-04: never present an old artifact as the current champion — a
    # strategy that classifies as legacy but carries no stamp gets the
    # legacy flag computed here (the file itself is stamped by
    # scripts/stamp_legacy_artifacts.py).
    if "legacy" not in payload:
        from ashare_model.artifact_versions import classify_strategy

        verdict = classify_strategy(payload)
        if verdict["legacy"]:
            payload = {
                **payload,
                "legacy": True,
                "legacy_reason": verdict["reasons"],
            }
    return payload


def get_sim_state() -> dict:
    """Transformed paper-trading portfolio state."""

    _, _, _, sim_config = _get_configs()
    if sim_config is None:
        return {}
    payload = _read_json(Path(sim_config.state_path)) or {}
    if not isinstance(payload, dict):
        return {}
    positions_raw = payload.get("positions") or {}
    positions = []
    market_value = 0.0
    for key, value in positions_raw.items():
        qty = int(value.get("quantity", 0))
        last_price = float(value.get("last_price", 0.0))
        row = {
            "ts_code": str(value.get("ts_code", key)),
            "name": value.get("name", ""),
            "quantity": qty,
            "available_quantity": int(value.get("available_quantity", 0)),
            "avg_cost": value.get("avg_cost"),
            "last_price": value.get("last_price"),
            "last_date": value.get("last_date"),
            "market_value": round(qty * last_price, 2),
        }
        market_value += qty * last_price
        positions.append(row)
    history = payload.get("equity_history") or []
    return {
        "initial_capital": payload.get("initial_capital"),
        "cash": payload.get("cash"),
        "trade_count": payload.get("trade_count", 0),
        "market_value": round(market_value, 2),
        "total_equity": round(float(payload.get("cash", 0)) + market_value, 2),
        "positions": positions,
        "equity_history": history,
    }


def get_sim_days() -> dict:
    """Dates for which order/trade paper trails exist, newest first."""

    _, _, _, sim_config = _get_configs()
    if sim_config is None:
        return {"total": 0, "dates": []}
    orders_dir = Path(sim_config.orders_dir)
    dates = sorted(
        (p.stem for p in orders_dir.glob("*.json") if p.stem[:4].isdigit()),
        reverse=True,
    )
    return {"total": len(dates), "dates": dates[:200]}


def get_sim_day(date: str) -> dict:
    _, _, _, sim_config = _get_configs()
    if sim_config is None or not date or not date[:4].isdigit():
        return {"date": date, "orders": [], "trades": []}
    orders = _read_json(Path(sim_config.orders_dir) / f"{date}.json") or []
    trades = _read_json(Path(sim_config.trades_dir) / f"{date}.json") or []
    return {
        "date": date,
        "orders": orders if isinstance(orders, list) else [],
        "trades": trades if isinstance(trades, list) else [],
    }


def get_data_status() -> dict:
    data_config, model_config, backtest_config, _ = _get_configs()
    if data_config is None:
        return {"ready": False, "reason": "config load failed"}
    db_info: dict = {"ready": False}
    try:
        with AshareDB(data_config.duckdb_path, read_only=True) as db:
            stocks = db.query(f"SELECT COUNT(*) AS n FROM {data_config.stocks_table}")
            daily = db.query(
                f"SELECT COUNT(*) AS n, MIN(trade_date) AS first_date, "
                f"MAX(trade_date) AS last_date FROM {data_config.daily_table}"
            )
            db_info = {
                "ready": True,
                "path": str(data_config.duckdb_path),
                "stocks": int(stocks.iloc[0]["n"]),
                "daily_rows": int(daily.iloc[0]["n"]),
                "first_trade_date": str(daily.iloc[0]["first_date"]),
                "last_trade_date": str(daily.iloc[0]["last_date"]),
            }
    except Exception as exc:  # noqa: BLE001
        db_info = {"ready": False, "reason": str(exc)}

    artifacts = []
    for path in (
        data_config.data_dir / "backtest_result.json",
        data_config.data_dir / "best_ashare_strategy.json",
        data_config.data_dir / "ashare_model.pt",
        data_config.data_dir / "sim_portfolio_state.json",
        data_config.data_dir / "ashare.duckdb",
    ):
        stat = _stat(path)
        artifacts.append(stat)

    config_summary = {
        "date_range": {
            "start": str(getattr(data_config, "start_date", "")),
            "end": str(getattr(data_config, "end_date", "")),
        },
        "universe": {
            "indexes": list(getattr(data_config, "index_codes", []) or []),
            "min_listed_sessions": getattr(
                data_config, "min_listed_sessions", None
            ),
        },
        "model": {
            "d_model": getattr(model_config, "d_model", None),
            "nhead": getattr(model_config, "nhead", None),
            "num_layers": getattr(model_config, "num_layers", None),
            "batch_size": getattr(model_config, "batch_size", None),
            "train_steps": getattr(model_config, "train_steps", None),
            "max_formula_len": getattr(model_config, "max_formula_len", None),
            "validation_fraction": getattr(model_config, "validation_fraction", None),
        },
        "backtest": {
            "train_end_date": getattr(backtest_config, "train_end_date", None),
            "top_n": getattr(backtest_config, "top_n", None),
            "single_weight_cap": getattr(backtest_config, "single_weight_cap", None),
            "commission_rate": getattr(backtest_config, "commission_rate", None),
            "stamp_tax_rate": getattr(backtest_config, "stamp_tax_rate", None),
            "slippage_rate": getattr(backtest_config, "slippage_rate", None),
            "initial_capital": getattr(backtest_config, "initial_capital", None),
            "benchmark": getattr(backtest_config, "benchmark", None),
        },
    }
    return {"db": db_info, "artifacts": artifacts, "config": config_summary}


def _stat(path: Path) -> dict:
    try:
        st = path.stat()
        return {
            "name": path.name,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "exists": True,
        }
    except OSError:
        return {"name": path.name, "size": 0, "mtime": "", "exists": False}


def _log_kind(name: str) -> str:
    lower = name.lower()
    for kind in ("train", "backtest", "sim", "sync", "pytest"):
        if lower.startswith(kind):
            return kind
    return "other"


def list_logs() -> list[dict]:
    items = []
    for directory in _LOG_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob("*"):
            if path.is_file() and path.suffix.lower() in {".log", ".txt"}:
                st = _stat(path)
                st["kind"] = _log_kind(path.name)
                items.append(st)
    items.sort(key=lambda x: (x["mtime"], x["name"]), reverse=True)
    return items


def read_log(name: str, tail: int = 1000) -> dict:
    """Return the tail of a log file. ``name`` is sanitized to a basename."""

    safe = Path(name).name
    for directory in _LOG_DIRS:
        candidate = directory / safe
        if candidate.is_file():
            try:
                content = _tail_text(candidate, tail)
                return {
                    "name": safe,
                    "size": candidate.stat().st_size,
                    "lines": content.count("\n") + (1 if content else 0),
                    "content": content,
                    "truncated": candidate.stat().st_size > _MAX_LOG_READ_BYTES,
                }
            except OSError as exc:
                return {"name": safe, "error": str(exc)}
    return {"name": safe, "error": "file not found"}


def _tail_text(path: Path, tail: int) -> str:
    """Read at most the last ``tail`` lines / 16MB of a UTF-8 log file."""

    size = path.stat().st_size
    read_size = min(size, _MAX_LOG_READ_BYTES)
    if read_size <= 0:
        return ""
    with open(path, "rb") as handle:
        if read_size < size:
            handle.seek(size - read_size)
            # Skip a possibly partial first line.
            handle.readline()
        blob = handle.read(read_size)
    text = blob.decode("utf-8", errors="replace")
    lines = text.splitlines()
    tail = max(1, int(tail))
    if len(lines) > tail:
        lines = lines[-tail:]
    return "\n".join(lines)


FEE_KEYS = (
    "commission_rate",
    "min_commission",
    "stamp_tax_rate",
    "transfer_fee_rate",
    "slippage_rate",
)


class SimConfigPatch(BaseModel):
    """Validated runtime config edits for the sim page config dialog.

    Fee keys belong to the shared ``backtest`` section (the single cost
    model used by backtests, training rewards and the paper trader);
    sim-only keys belong to the ``sim`` section. A ``None`` value removes
    the key from the runtime overrides file so the YAML baseline applies
    again.
    """

    initial_capital: float | None = Field(default=None, gt=0)
    max_positions: int | None = Field(default=None, ge=1, le=500)
    single_weight_cap: float | None = Field(default=None, gt=0, le=1)
    commission_rate: float | None = Field(default=None, ge=0, lt=1)
    min_commission: float | None = Field(default=None, ge=0)
    stamp_tax_rate: float | None = Field(default=None, ge=0, lt=1)
    transfer_fee_rate: float | None = Field(default=None, ge=0, lt=1)
    slippage_rate: float | None = Field(default=None, ge=0, lt=1)


def _config_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "config" / "ashare_config.yaml",
        root / "config" / RUNTIME_OVERRIDES_FILENAME,
    )


def _read_overrides(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001 - dashboard must survive a bad file.
        return {}


def _write_overrides(path: Path, overrides: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.yaml")
    tmp.write_text(
        yaml.safe_dump(overrides, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_sim_config(patch: SimConfigPatch, root: Path = ROOT) -> dict:
    """Apply a validated config patch to the runtime overrides file."""

    config_path, overrides_path = _config_paths(root)
    if not config_path.exists():
        return {"ok": False, "reason": f"config file not found: {config_path}"}

    overrides = _read_overrides(overrides_path)
    patch_map = {
        "sim": {
            "initial_capital": patch.initial_capital,
            "max_positions": patch.max_positions,
            "single_weight_cap": patch.single_weight_cap,
        },
        "backtest": {
            "initial_capital": patch.initial_capital,
            "top_n": patch.max_positions,
            "single_weight_cap": patch.single_weight_cap,
            **{key: getattr(patch, key) for key in FEE_KEYS},
        },
    }
    for section, mapping in patch_map.items():
        target = overrides.setdefault(section, {})
        for key, value in mapping.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
    for section in tuple(overrides):
        if section in ("sim", "backtest") and not overrides[section]:
            overrides.pop(section, None)

    try:
        _write_overrides(overrides_path, overrides)
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}

    result = get_sim_config(root=root)
    result["ok"] = True
    return result


def get_sim_config(root: Path = ROOT) -> dict:
    """Effective sim/fee config (YAML baseline + runtime overrides) and the
    initial-capital reset pendency.

    Defensive degradation (AGENTS.md §9 UI clause): when the YAML baseline
    cannot be loaded, factory defaults are still returned so the dashboard
    stays alive, but the response is explicitly marked
    ``config_degraded=True`` with ``degraded_reason`` — it must not be
    presented as the healthy effective configuration.
    """

    config_path, overrides_path = _config_paths(root)
    degraded_reason: str | None = None
    try:
        raw = load_config(config_path, project_root=root)
    except Exception as exc:  # noqa: BLE001 - dashboard must survive bad config,
        # but the degradation must be visible, not silent.
        raw = {}
        degraded_reason = f"{type(exc).__name__}: {exc}"
    sim = make_sim_config(raw, root)
    backtest = make_backtest_config(raw)

    effective = {
        "initial_capital": sim.initial_capital,
        "max_positions": sim.max_positions,
        "single_weight_cap": sim.single_weight_cap,
        "commission_rate": backtest.commission_rate,
        "min_commission": backtest.min_commission,
        "stamp_tax_rate": backtest.stamp_tax_rate,
        "transfer_fee_rate": backtest.transfer_fee_rate,
        "slippage_rate": backtest.slippage_rate,
    }
    state = _read_json(Path(sim.state_path)) or {}
    state_initial = state.get("initial_capital") if isinstance(state, dict) else None
    pending_reset = state_initial is None or float(state_initial) != float(
        effective["initial_capital"]
    )
    mismatches = execution_config_mismatches(backtest, sim)
    return {
        "effective": effective,
        "config_degraded": degraded_reason is not None,
        "degraded_reason": degraded_reason,
        "overrides_path": str(overrides_path),
        "overrides": _read_overrides(overrides_path),
        "state_initial_capital": state_initial,
        "pending_reset": pending_reset,
        "execution_config_consistent": not mismatches,
        "execution_config_mismatches": {
            key: {"backtest": left, "sim": right}
            for key, (left, right) in mismatches.items()
        },
    }


class SimStartRequest(BaseModel):
    """Start parameters. ``reset`` archives and clears the current state;
    otherwise the runner resumes from ``last_exec_date`` (or replays when no
    state exists). Date strings accept YYYY-MM-DD or YYYYMMDD."""

    reset: bool = False
    start: str | None = Field(default=None, pattern=r"^\d{4}-?\d{2}-?\d{2}$")
    end: str | None = Field(default=None, pattern=r"^\d{4}-?\d{2}-?\d{2}$")


def _job_manager():
    from ashare_trading.manager import SimJobManager

    _, _, _, sim_config = _get_configs()
    return SimJobManager(ROOT, sim_config)


def sim_start(req: SimStartRequest) -> dict:
    from ashare_data.gates import ProductionGateRunner
    from ashare_trading.manager import RunConflictError

    data_config, _, backtest_config, sim_config = _get_configs()
    if data_config is None:
        return {"ok": False, "reason": "config load failed"}
    try:
        validate_execution_config(backtest_config, sim_config)
        ProductionGateRunner(data_config).require_production()
    except (ValueError, UniverseContractError) as exc:
        return {"ok": False, "reason": str(exc)}
    strategy = data_config.data_dir / "best_ashare_strategy.json"
    if not strategy.exists():
        return {
            "ok": False,
            "reason": f"strategy file missing: {strategy} (train first)",
        }
    try:
        return _job_manager().start(
            reset=req.reset, start_date=req.start, end_date=req.end
        )
    except RunConflictError as exc:
        return {"ok": False, "conflict": True, "reason": str(exc)}
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}


def sim_stop_run() -> dict:
    try:
        return _job_manager().stop()
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}


def sim_reset_run() -> dict:
    from ashare_trading.manager import RunConflictError

    try:
        return _job_manager().reset()
    except RunConflictError as exc:
        return {"ok": False, "conflict": True, "reason": str(exc)}
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}


def sim_status() -> dict:
    try:
        return _job_manager().status()
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}
