"""Streamlit research and paper-trading dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_sim_config,
)
from ashare_data.db import AshareDB

from .data_service import (
    load_backtest_result,
    load_data_status,
    load_sim_state,
)
from .visualizer import equity_figure, factor_bar_figure


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_configs():
    root = _root()
    raw = load_config(None, project_root=root)
    return (
        make_data_config(raw, root),
        make_model_config(raw),
        make_backtest_config(raw),
        make_sim_config(raw, root),
    )


def main() -> None:
    st.set_page_config(page_title="AlphaGPT A股", layout="wide")
    st.title("AlphaGPT 纯 A 股多因子模拟盘")
    data_config, _, backtest_config, sim_config = _load_configs()

    tab_overview, tab_selection, tab_backtest, tab_sim, tab_data = st.tabs(
        ["概览", "选股", "回测", "模拟盘", "数据状态"]
    )

    with tab_overview:
        st.header("概览")
        result = load_backtest_result(data_config)
        if result and "metrics" in result:
            metrics = result["metrics"]
            cols = st.columns(4)
            cols[0].metric("累计收益", f"{metrics.get('total_return', 0):.2%}")
            cols[1].metric("年化收益", f"{metrics.get('annual_return', 0):.2%}")
            cols[2].metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
            cols[3].metric("Sortino", f"{metrics.get('sortino', 0):.2f}")
            st.plotly_chart(
                equity_figure(
                    result.get("dates", []),
                    result.get("equity_curve", []),
                    "策略净值",
                    benchmark_equity=result.get("benchmark_equity"),
                    benchmark_name=str(result.get("benchmark", "基准")),
                ),
                width="stretch",
            )
        else:
            st.info("暂无回测结果，请先运行 python -m ashare_model.backtest")

    with tab_selection:
        st.header("选股")
        result = load_backtest_result(data_config)
        if result and result.get("positions"):
            latest = result["positions"][-1]
            st.subheader(f"信号日 {latest.get('signal_date')}")
            st.write(f"公式：{result.get('formula_text', 'N/A')}")
            st.dataframe(pd.DataFrame({"股票": latest["ts_codes"], "权重": latest["weights"]}))
        else:
            st.info("暂无持仓信号")

    with tab_backtest:
        st.header("回测")
        result = load_backtest_result(data_config)
        if result and "equity_curve" in result:
            st.plotly_chart(
                equity_figure(
                    result.get("dates", []),
                    result.get("equity_curve", []),
                    "回测净值",
                    benchmark_equity=result.get("benchmark_equity"),
                    benchmark_name=str(result.get("benchmark", "基准")),
                ),
                width="stretch",
            )
            st.json(result.get("metrics", {}))
        else:
            st.info("暂无回测结果")

    with tab_sim:
        st.header("模拟盘")
        state = load_sim_state(sim_config)
        if state:
            st.metric("可用资金", f"{state.get('cash', 0):,.2f}")
            st.metric("成交笔数", state.get("trade_count", 0))
            positions = state.get("positions", {})
            if positions:
                rows = [
                    {
                        "ts_code": v.get("ts_code"),
                        "name": v.get("name"),
                        "quantity": v.get("quantity"),
                        "available_quantity": v.get("available_quantity"),
                        "avg_cost": v.get("avg_cost"),
                        "last_price": v.get("last_price"),
                    }
                    for v in positions.values()
                ]
                st.dataframe(pd.DataFrame(rows))
            if state.get("equity_history"):
                hist = pd.DataFrame(state["equity_history"])
                st.plotly_chart(
                    equity_figure(
                        hist["trade_date"].astype(str).tolist(),
                        hist["equity"].astype(float).tolist(),
                        "模拟盘资金曲线",
                    ),
                    width="stretch",
                )
        else:
            st.info("暂无模拟盘状态")

        if st.button("写入紧急停止信号"):
            Path(sim_config.stop_signal_path).write_text("STOP", encoding="utf-8")
            st.success("已写入 STOP_SIGNAL")

    with tab_data:
        st.header("数据状态")
        status = load_data_status(data_config)
        st.json(status)


if __name__ == "__main__":
    main()
