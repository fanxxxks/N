"""P37 R02 six-cell portfolio/cost matrix tests (round 1, period frozen).

Executable projection of ``docs/p37_r02_portfolio_cost_matrix_draft.md``
(APPROVED via user direct instruction 2026-09-19 work-package 4 「把 R02 首轮
缩成六格，先固定调仓周期」).  RED-first: every test below asserts an
independent invariant from the contract — the frozen grid table with its
coupled per-cell params, the exact-five-field cell variant, the hand-computed
continuous-vs-lot gap fields, the four-way adjudication with per-top_n
holdings floors, the grid adjacency definition, and the end-to-end lifecycle
chain on the shared mini dual-DB fixture (imported from the sibling p34
module — single fixture source, the same pattern test_p34_r01_report_wp3
uses).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ashare_data.config import BacktestConfig

from test_p34_r01_baseline import (
    _contract_sha as _p34_contract_sha,  # noqa: F401  (fixture lineage check)
    _mini_dates,
    gate_run_asof,  # noqa: F401  (pytest fixture, re-exported)
    mini_env,  # noqa: F401  (pytest fixture, re-exported)
)

_ROOT = Path(__file__).resolve().parents[1]

LOT = 100


def _contract_sha() -> str:
    """sha256 of the committed p37 contract (the C5 self-verification seed)."""

    digest = hashlib.sha256()
    with (_ROOT / "docs" / "p37_r02_portfolio_cost_matrix_draft.md").open(
        "rb"
    ) as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- W1: the frozen six-cell grid --------------------------------------------


def test_w1_grid_frozen_definition():
    from ashare_model.r02_matrix import R02_GRID, R02_REFERENCE_CELL

    assert len(R02_GRID) == 6
    by_name = {cell.name: cell for cell in R02_GRID}
    assert set(by_name) == {
        "n10_m1000", "n10_m2500", "n10_m5000",
        "n20_m1000", "n20_m2500", "n20_m5000",
    }
    for cell in R02_GRID:
        # Coupled-param invariants (p37 §3.2 rules 1-3, independent of any
        # measurement): buy_rank = top_n; single_weight_cap = 1/top_n so
        # full investment is representable; sell_rank keeps the 1.5x
        # buffer ratio of r1_medium_a.
        assert cell.buy_rank == cell.top_n
        assert cell.single_weight_cap == pytest.approx(1.0 / cell.top_n)
        assert cell.sell_rank == math.ceil(1.5 * cell.top_n)
        assert cell.min_trade_amount in (1000.0, 2500.0, 5000.0)
    # The reference cell is the r1_medium_a backtest block verbatim.
    ref = by_name[R02_REFERENCE_CELL]
    assert (ref.top_n, ref.buy_rank, ref.sell_rank) == (20, 20, 30)
    assert ref.single_weight_cap == pytest.approx(0.05)
    assert ref.min_trade_amount == 5000.0
    # Both axes span the preregistered levels.
    assert {c.top_n for c in R02_GRID} == {10, 20}
    assert {c.min_trade_amount for c in R02_GRID} == {1000.0, 2500.0, 5000.0}


# --- W2: the cell variant replaces exactly five fields -----------------------


def test_w2_cell_variant_replaces_exactly_five_fields():
    from ashare_model.r02_matrix import cell_variant

    base = BacktestConfig(
        rebalance_frequency="every_10_days",
        target_horizon=10,
        initial_capital=100000.0,
        top_n=20,
        portfolio_method="equal_weight",
        buy_rank=20,
        sell_rank=30,
        single_weight_cap=0.05,
        min_trade_amount=5000.0,
        target_weight_change_threshold=0.01,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    replaced = {
        "top_n",
        "buy_rank",
        "sell_rank",
        "single_weight_cap",
        "min_trade_amount",
    }
    for name in (
        "n10_m1000", "n10_m2500", "n10_m5000",
        "n20_m1000", "n20_m2500", "n20_m5000",
    ):
        variant = cell_variant(base, name)
        assert isinstance(variant, BacktestConfig)
        for field in dataclasses.fields(BacktestConfig):
            key = field.name
            if key in replaced:
                continue
            assert getattr(variant, key) == getattr(base, key), (
                f"cell {name}: field {key} drifted"
            )
    # The n10 coupling lands verbatim.
    v = cell_variant(base, "n10_m2500")
    assert (v.top_n, v.buy_rank, v.sell_rank) == (10, 10, 15)
    assert v.single_weight_cap == pytest.approx(0.10)
    assert v.min_trade_amount == 2500.0
    # The reference cell round-trips to the base values.
    v = cell_variant(base, "n20_m5000")
    assert (v.top_n, v.buy_rank, v.sell_rank) == (20, 20, 30)
    assert v.single_weight_cap == pytest.approx(0.05)
    assert v.min_trade_amount == 5000.0
    with pytest.raises(ValueError, match="unknown cell"):
        cell_variant(base, "n30_m1000")


# --- W3: the continuous-vs-lot gap, hand computed -----------------------------


def _p37_bt_config(**over) -> BacktestConfig:
    fields = dict(
        portfolio_method="equal_weight",
        buy_rank=1,
        sell_rank=1,
        top_n=1,
        single_weight_cap=1.0,
        initial_capital=100_000.0,
        min_trade_amount=None,
        target_weight_change_threshold=0.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    fields.update(over)
    return BacktestConfig(**fields)


def _p37_raw_cache(open_: np.ndarray) -> dict:
    return {
        "open": open_,
        "volume": np.ones_like(open_),
        "adjusted_open": open_.copy(),
        "up_limit": open_ * 10.0,
        "down_limit": open_ * 0.1,
        "suspended": np.zeros_like(open_),
    }


def _p37_run(open_: np.ndarray, rebalance_mask: np.ndarray, **config_over):
    from ashare_model.backtest import AshareBacktestEngine

    n_stocks, n_dates = open_.shape
    engine = AshareBacktestEngine(_p37_bt_config(**config_over))
    factors = np.vstack([
        np.full(n_dates, 1.0),
        np.full(n_dates, 0.5),
    ])
    return engine.run(
        factors,
        _p37_raw_cache(open_),
        [f"{i:06d}.SZ" for i in range(n_stocks)],
        [f"202601{d:02d}" for d in range(1, n_dates + 1)],
        universe_mask=np.ones_like(open_, dtype=bool),
        rebalance_mask=rebalance_mask,
    )


def test_w3_lot_gap_fields_hand_computed():
    """One due session, one name, lot floor leaves 1,000 CNY uninvested:
    target weight 1.0 -> 3,000 shares of a 33.0 CNY open = 99,000 notional
    = 0.99 realized.  Full-L1 gap 0.01, invested gap 0.01, one initial
    funding session with 1 planned and 1 filled order."""

    from ashare_model.eval_metrics import lot_gap_fields

    open_ = np.array([[33.0] * 5, [20.0] * 5])
    rebalance_mask = np.array([True, False, False, False, False])
    result = _p37_run(open_, rebalance_mask)
    fields = lot_gap_fields(result)
    assert fields["lotgap_l1_mean_due"] == pytest.approx(0.01, rel=1e-9)
    assert fields["lotgap_l1_max_due"] == pytest.approx(0.01, rel=1e-9)
    assert fields["lotgap_invested_mean_due"] == pytest.approx(
        0.01, rel=1e-9
    )
    assert fields["funnel_initial_funding_sessions"] == 1
    assert fields["funnel_initial_funding_planned_orders"] == 1
    assert fields["funnel_initial_funding_filled_orders"] == 1
    # The engine's own economics are untouched (additive observability):
    # 99,000 invested at a flat 33.0 open earns nothing through session 4.
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)


def test_w3_lot_gap_fields_null_without_due_sessions():
    from ashare_model.eval_metrics import lot_gap_fields

    open_ = np.array([[33.0] * 5, [20.0] * 5])
    rebalance_mask = np.array([False] * 5)
    result = _p37_run(open_, rebalance_mask)
    fields = lot_gap_fields(result)
    assert fields["lotgap_l1_mean_due"] is None
    assert fields["lotgap_l1_max_due"] is None
    assert fields["lotgap_invested_mean_due"] is None
    assert fields["funnel_initial_funding_sessions"] == 0


# --- W4: the frozen four-way adjudication -------------------------------------


def _arm_summary(
    *,
    excess_1x=0.05,
    ir_1x=0.2,
    invested=0.9,
    holdings=12.0,
    failed_folds=0,
    positive_folds=4,
    gross_excess_0x=0.08,
    gross_ir_0x=0.3,
):
    return {
        "mean_net_excess_1x": excess_1x,
        "active_ir_1x": ir_1x,
        "invested_ratio_mean_1x": invested,
        "mean_holdings_1x": holdings,
        "failed_folds": failed_folds,
        "positive_folds_1x": positive_folds,
        "stitched": {
            "0x": {"excess_return": gross_excess_0x, "oos_active_ir": gross_ir_0x},
        },
    }


def _cells(cell_arms: dict[str, dict[str, dict]]) -> dict:
    from ashare_model.r02_matrix import R02_GRID

    params = {cell.name: cell for cell in R02_GRID}
    return {
        name: {
            "top_n": params[name].top_n,
            "params": {
                "top_n": params[name].top_n,
                "buy_rank": params[name].buy_rank,
                "sell_rank": params[name].sell_rank,
                "single_weight_cap": params[name].single_weight_cap,
                "min_trade_amount": params[name].min_trade_amount,
            },
            "arms": arms,
        }
        for name, arms in cell_arms.items()
    }


def _full_panel(viable_cell: str | None, arm=None) -> dict:
    """All six cells; every non-viable cell carries a dead arm (and no
    gross info unless stated), the viable one carries a live arm."""

    from ashare_model.r02_matrix import R02_GRID

    out = {}
    for cell in R02_GRID:
        if cell.name == viable_cell:
            out[cell.name] = {"baseline:X": arm or _arm_summary()}
        else:
            out[cell.name] = {
                "baseline:X": _arm_summary(
                    excess_1x=-0.01,
                    ir_1x=-0.05,
                    invested=0.2,
                    holdings=float(cell.top_n),
                    gross_excess_0x=-0.02,
                    gross_ir_0x=-0.1,
                )
            }
    return _cells(out)


def test_w4_stable_region_found():
    from ashare_model.r02_matrix import decide_r02_adjudication

    cells = _full_panel(None)
    cells["n20_m1000"]["arms"]["baseline:X"] = _arm_summary()
    cells["n20_m2500"]["arms"]["baseline:X"] = _arm_summary()
    verdict = decide_r02_adjudication(cells)
    assert verdict["verdict"] == "stable_region_found"
    assert set(verdict["viable_cells"]) == {"n20_m1000", "n20_m2500"}
    assert ["n20_m1000", "n20_m2500"] in [
        list(pair) for pair in verdict["stable_pairs"]
    ]


def test_w4_isolated_single_winner_is_unstable():
    from ashare_model.r02_matrix import decide_r02_adjudication

    verdict = decide_r02_adjudication(_full_panel("n10_m2500"))
    assert verdict["verdict"] == "isolated_single_winner"
    assert verdict["isolated_cell"] == "n10_m2500"
    assert verdict["stable_pairs"] == []
    # The instability note is recorded verbatim: a lone cell never
    # justifies a parameter choice.
    assert "不稳定" in verdict["note"] or "unstable" in verdict["note"].lower()


def test_w4_gross_info_execution_blocked():
    from ashare_model.r02_matrix import decide_r02_adjudication

    cells = _full_panel(None)
    # No viable cell, but one arm keeps gross space at 0x.
    cells["n20_m1000"]["arms"]["baseline:X"] = _arm_summary(
        excess_1x=-0.02, ir_1x=-0.1, invested=0.2,
        gross_excess_0x=0.15, gross_ir_0x=0.4,
    )
    verdict = decide_r02_adjudication(cells)
    assert verdict["verdict"] == "gross_info_execution_blocked"
    assert "n20_m1000::baseline:X" in verdict["gross_info_arms"]


def test_w4_panel_no_reliable_info_pauses():
    from ashare_model.r02_matrix import decide_r02_adjudication

    verdict = decide_r02_adjudication(_full_panel(None))
    assert verdict["verdict"] == "panel_no_reliable_info"
    assert verdict["gross_info_arms"] == []


def test_w4_holdings_floor_scales_with_top_n():
    """holdings_mean = 7 clears the n10 floor (>= 5) but not the n20
    floor (>= 10) — the same evidence is viable in one cell and not the
    other."""

    from ashare_model.r02_matrix import decide_r02_adjudication

    cells = _full_panel(None)
    cells["n10_m1000"]["arms"]["baseline:X"] = _arm_summary(holdings=7.0)
    verdict = decide_r02_adjudication(cells)
    assert verdict["viable_cells"] == ["n10_m1000"]

    cells = _full_panel(None)
    cells["n20_m1000"]["arms"]["baseline:X"] = _arm_summary(holdings=7.0)
    verdict = decide_r02_adjudication(cells)
    assert verdict["viable_cells"] == []
    flagged = verdict["cells"]["n20_m1000"]["arms"]["baseline:X"]["flags"]
    assert "thin_book" in flagged


def test_w4_low_activity_flagged():
    from ashare_model.r02_matrix import decide_r02_adjudication

    cells = _full_panel(None)
    cells["n20_m1000"]["arms"]["baseline:X"] = _arm_summary(invested=0.3)
    verdict = decide_r02_adjudication(cells)
    assert verdict["viable_cells"] == []
    flagged = verdict["cells"]["n20_m1000"]["arms"]["baseline:X"]["flags"]
    assert "low_activity" in flagged


def test_w4_failed_folds_make_arm_unreliable():
    from ashare_model.r02_matrix import decide_r02_adjudication

    cells = _full_panel(None)
    cells["n20_m1000"]["arms"]["baseline:X"] = _arm_summary(failed_folds=2)
    verdict = decide_r02_adjudication(cells)
    assert verdict["viable_cells"] == []
    flagged = verdict["cells"]["n20_m1000"]["arms"]["baseline:X"]["flags"]
    assert "unreliable" in flagged


# --- W5: grid adjacency -------------------------------------------------------


def test_w5_adjacency_definition():
    from ashare_model.r02_matrix import cell_adjacent

    assert cell_adjacent("n10_m1000", "n10_m2500") is True
    assert cell_adjacent("n10_m2500", "n10_m5000") is True
    assert cell_adjacent("n10_m1000", "n20_m1000") is True
    assert cell_adjacent("n20_m2500", "n20_m5000") is True
    # Diagonal / skip / self are NOT adjacent.
    assert cell_adjacent("n10_m1000", "n20_m2500") is False
    assert cell_adjacent("n10_m1000", "n10_m5000") is False
    assert cell_adjacent("n10_m1000", "n20_m5000") is False
    assert cell_adjacent("n10_m1000", "n10_m1000") is False


# --- W6: frozen constants -----------------------------------------------------


def test_w6_constants_frozen():
    from ashare_model.r02_matrix import (
        R02_COST_LEVELS,
        R02_REFERENCE_CELL,
        R02_REPORT_SCHEMA_VERSION,
    )

    assert R02_COST_LEVELS == ("0x", "1x")
    assert R02_REFERENCE_CELL == "n20_m5000"
    assert R02_REPORT_SCHEMA_VERSION == 1


# --- W7: end-to-end through the real lifecycle chain --------------------------


def test_w7_contract_sha_mismatch_refused(mini_env, gate_run_asof):
    import pytest as _pytest

    import ashare_model.r02_matrix as r02

    tmp_path, _data_config, cfg_path = mini_env
    gate_run_asof(_mini_dates()[-1])
    with _pytest.raises(r02.R02Refusal, match="contract sha256 mismatch"):
        r02.run_r02_matrix(
            cfg_path,
            contract_sha256="0" * 64,
            out_root=tmp_path / "r02_out_sha",
            require_clean_tree=False,
            min_eligible=3,
        )


def test_w7_budget_exhaustion_marks_failed(mini_env, gate_run_asof):
    import ashare_model.r02_matrix as r02
    from ashare_model.lifecycle import replay_run
    from ashare_model.run_store import RunStore

    tmp_path, data_config, cfg_path = mini_env
    gate_run_asof(_mini_dates()[-1])
    with pytest.raises(r02.R02Refusal, match="wall-clock budget"):
        r02.run_r02_matrix(
            cfg_path,
            contract_sha256=_contract_sha(),
            out_root=tmp_path / "r02_out_budget",
            require_clean_tree=False,
            min_eligible=3,
            wall_clock_budget_s=0.0,
        )
    store = RunStore(data_config.data_dir)
    spec_dir = next(
        p for p in (data_config.data_dir / "runs").iterdir() if p.is_dir()
    )
    run_dir = next(
        p
        for p in spec_dir.iterdir()
        if p.is_dir() and p.name != "artifacts"
    )
    assert replay_run(store.root, spec_dir.name, run_dir.name) == "FAILED"


def test_w7_data_gate_refusal_rejected(mini_env, gate_run_asof, monkeypatch):
    from ashare_model.data_loader import AshareDataLoader
    from ashare_model.lifecycle import replay_run
    from ashare_model.run_store import RunStore

    import ashare_model.r02_matrix as r02

    tmp_path, data_config, cfg_path = mini_env
    gate_run_asof(_mini_dates()[-1])

    def degraded_status(self):
        return {
            "status": "degraded",
            "reasons": ["price_semantics_degraded=['000001.SZ'] (test)"],
        }

    monkeypatch.setattr(AshareDataLoader, "loader_frame_status", degraded_status)
    with pytest.raises(r02.R02Refusal, match="data qualification refused"):
        r02.run_r02_matrix(
            cfg_path,
            contract_sha256=_contract_sha(),
            out_root=tmp_path / "r02_out_rej",
            require_clean_tree=False,
            min_eligible=3,
        )
    store = RunStore(data_config.data_dir)
    spec_dir = sorted(
        p for p in (data_config.data_dir / "runs").iterdir() if p.is_dir()
    )[-1]
    run_dir = next(
        p
        for p in spec_dir.iterdir()
        if p.is_dir() and p.name != "artifacts"
    )
    assert replay_run(store.root, spec_dir.name, run_dir.name) == "REJECTED"


def test_w7_end_to_end_chain_verdict_consistent(mini_env, gate_run_asof):
    """The full chain on the shared mini fixture through the REAL frozen
    rules (relaxed canary thresholds via the explicit test seam): the
    verdict comes from the real adjudicator on real engine rows, and the
    administrative state must match the verdict mapping (verdict 1/2 park
    non-terminal at DATA_QUALIFIED; verdict 3/4 retire).  Structure is
    asserted deterministically: all six cells x all five arms x both cost
    levels present, the benchmark as the cost-free reference, the cash
    control at the reference cell only, the frozen directions recorded,
    json-safe rows, and the additive lot-gap fields wired into every
    cost-bearing row."""

    import ashare_model.r02_matrix as r02
    from ashare_model.lifecycle import replay_run
    from ashare_model.run_store import RunStore
    from ashare_data.config import make_data_config

    tmp_path, _data_config, cfg_path = mini_env
    gate_run_asof(_mini_dates()[-1])
    summary = r02.run_r02_matrix(
        cfg_path,
        contract_sha256=_contract_sha(),
        out_root=tmp_path / "r02_out",
        require_clean_tree=False,
        min_eligible=3,
        adjudication_rules={
            "min_positive_folds": 1,
            "min_invested_ratio": 0.3,
            "max_failed_folds": 1,
        },
    )
    verdict = summary["grid_adjudication"]["verdict"]
    assert verdict in {
        "stable_region_found",
        "isolated_single_winner",
        "gross_info_execution_blocked",
        "panel_no_reliable_info",
    }
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, tmp_path)
    store = RunStore(data_config.data_dir)
    state = replay_run(store.root, summary["spec_id"], summary["run_id"])
    if verdict in ("stable_region_found", "isolated_single_winner"):
        assert state == "DATA_QUALIFIED"
    else:
        assert state == "RETIRED"

    report_path = tmp_path / "r02_out" / "r02_matrix_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["r02_report_schema_version"] == 1
    assert payload["provenance"]["contract_sha256"] == _contract_sha()
    assert payload["provenance"]["dataset_id"] == summary["dataset_id"]
    assert payload["provenance"]["consumed_semantic_evaluations"] == 0
    assert payload["grid_adjudication"]["verdict"] == verdict
    # The frozen grid lands verbatim in the provenance.
    assert {
        cell["name"] for cell in payload["provenance"]["grid"]
    } == {
        "n10_m1000", "n10_m2500", "n10_m5000",
        "n20_m1000", "n20_m2500", "n20_m5000",
    }
    rows = payload["rows"]
    cells_in_rows = {
        row.get("cell") for row in rows if row.get("cell")
    }
    assert cells_in_rows == {
        "n10_m1000", "n10_m2500", "n10_m5000",
        "n20_m1000", "n20_m2500", "n20_m5000",
    }
    # Benchmark: the cost-free reference once per fold, outside the ladder.
    bench = [row for row in rows if row["candidate"] == "benchmark:equal_weight"]
    assert len(bench) == 1
    assert bench[0]["cost_level"] == "ref"
    # Cash control: only at the reference cell, 1x.
    cash = [row for row in rows if row["candidate"] == "control:cash"]
    assert len(cash) == 1
    assert cash[0]["cell"] == "n20_m5000"
    assert cash[0]["cost_level"] == "1x"
    # Every cost-bearing (cell, arm) pair at both levels — full-panel
    # reporting, no trimming to the best row.
    arms = {
        "baseline:MOMENTUM_20", "baseline:RSQ_60", "baseline:REVERSAL_60",
        "combo:all3", "combo:momentum_reversal",
    }
    panel_keys = {
        (row["cell"], row["candidate"].split("::", 1)[1], row["cost_level"])
        for row in rows
        if "::" in row["candidate"]
    }
    assert len(panel_keys) == 6 * len(arms) * 2
    assert {row["cost_level"] for row in rows} == {"0x", "1x", "ref"}
    # The additive lot-gap fields are wired into evaluate_signal rows.
    cost_bearing = [row for row in rows if "::" in row["candidate"]]
    assert cost_bearing
    for row in cost_bearing:
        assert "lotgap_l1_mean_due" in row
        if row["funnel_due_sessions"] > 0:
            assert row["lotgap_l1_mean_due"] is not None
    # Frozen directions with their train-window evidence, per fold.
    directions = payload["provenance"]["directions_by_fold"]
    assert len(directions) == 1
    for arm in (
        "MOMENTUM_20", "RSQ_60", "REVERSAL_60",
    ):
        entry = directions[list(directions)[0]][arm]
        assert entry["direction"] in (-1, 1)
        assert "val_reward" in entry
