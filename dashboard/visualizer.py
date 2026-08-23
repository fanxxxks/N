"""Plotly chart builders."""

from __future__ import annotations

import plotly.graph_objects as go


def equity_figure(
    dates,
    equity,
    title: str,
    benchmark_equity=None,
    benchmark_name: str = "基准",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines", name="净值"))
    if benchmark_equity:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark_equity,
                mode="lines",
                name=benchmark_name,
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(title=title, xaxis_title="日期", yaxis_title="净值", height=420)
    return fig
