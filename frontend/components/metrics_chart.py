"""
Evaluation metrics visualizations.
All charts use Plotly (already available via streamlit).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# Catppuccin Mocha palette for charts
_COLORS = ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7", "#fab387", "#94e2d5", "#f38ba8"]
_BG = "#1e1e2e"
_PAPER_BG = "#181825"
_GRID = "#313244"
_TEXT = "#cdd6f4"

_METRIC_LABELS = {
    "map": "MAP",
    "recall": "Recall@100",
    "precision_at_10": "P@10",
    "ndcg_at_10": "nDCG@10",
}


def _base_layout(**kwargs) -> dict:
    return dict(
        plot_bgcolor=_BG,
        paper_bgcolor=_PAPER_BG,
        font=dict(color=_TEXT, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=_GRID, linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, linecolor=_GRID),
        margin=dict(l=40, r=20, t=40, b=40),
        **kwargs,
    )


def _no_plotly() -> None:
    st.warning("Plotly is not installed. Run `pip install plotly` to enable charts.")


def render_metrics_bar_chart(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Grouped bar chart: x = metric name, groups = models.
    metrics: {"TF-IDF": {"map": 0.72, "recall": 0.85, ...}, "BM25": {...}}
    """
    if not _PLOTLY:
        return _no_plotly()
    if not metrics:
        st.info("No metrics available.")
        return

    metric_keys = list(_METRIC_LABELS.keys())
    fig = go.Figure()
    for idx, (model_name, vals) in enumerate(metrics.items()):
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=[_METRIC_LABELS.get(k, k) for k in metric_keys],
                y=[vals.get(k, 0.0) for k in metric_keys],
                marker_color=_COLORS[idx % len(_COLORS)],
            )
        )
    fig.update_layout(
        barmode="group",
        title="Model Comparison — IR Metrics",
        yaxis_range=[0, 1],
        **_base_layout(),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(metrics: Dict[str, Dict[str, float]]) -> None:
    """Spider/radar chart: one trace per model."""
    if not _PLOTLY:
        return _no_plotly()
    if not metrics:
        return

    categories = [_METRIC_LABELS.get(k, k) for k in _METRIC_LABELS]
    fig = go.Figure()
    for idx, (model_name, vals) in enumerate(metrics.items()):
        values = [vals.get(k, 0.0) for k in _METRIC_LABELS]
        values.append(values[0])
        
        color_hex = _COLORS[idx % len(_COLORS)]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        fillcolor = f"rgba({r},{g},{b},0.15)"
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=model_name,
                line_color=color_hex,
                fillcolor=fillcolor,
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=_GRID, color=_TEXT),
            angularaxis=dict(gridcolor=_GRID, color=_TEXT),
        ),
        title="Metric Radar",
        paper_bgcolor=_PAPER_BG,
        font=dict(color=_TEXT, family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_precision_at_k_curve(
    pk_data: Dict[str, Dict[int, float]],
    k_values: Optional[List[int]] = None,
) -> None:
    """
    Line chart: Precision@K for each model.
    pk_data: {"TF-IDF": {1: 0.9, 3: 0.8, 5: 0.75, 10: 0.6, ...}, ...}
    """
    if not _PLOTLY:
        return _no_plotly()
    if not pk_data:
        return

    k_values = k_values or [1, 3, 5, 10, 20, 50, 100]
    fig = go.Figure()
    
    all_values = []
    for idx, (model_name, pk_vals) in enumerate(pk_data.items()):
        y_values = [pk_vals.get(k, None) for k in k_values]
        all_values.extend([v for v in y_values if v is not None])
        
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=y_values,
                mode="lines+markers",
                name=model_name,
                line=dict(color=_COLORS[idx % len(_COLORS)], width=3),
                marker=dict(size=8),
                connectgaps=True,
            )
        )
    
    # Auto-scale y-axis based on actual values with some padding
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        padding = (max_val - min_val) * 0.15
        y_min = max(0, min_val - padding)
        y_max = min(1, max_val + padding)
    else:
        y_min, y_max = 0, 1
    
    fig.update_layout(
        title="Precision@K Curve",
        xaxis_title="K (Number of Retrieved Documents)",
        yaxis_title="Precision",
        yaxis_range=[y_min, y_max],
        height=550,
        **_base_layout(),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_refinement_comparison_chart(
    before: Dict[str, float],
    after: Dict[str, float],
) -> None:
    """Grouped bars: metrics before vs after query refinement for one model."""
    if not _PLOTLY:
        return _no_plotly()

    metric_keys = list(_METRIC_LABELS.keys())
    labels = [_METRIC_LABELS.get(k, k) for k in metric_keys]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Before Refinement",
                x=labels,
                y=[before.get(k, 0.0) for k in metric_keys],
                marker_color=_COLORS[1],
            ),
            go.Bar(
                name="After Refinement",
                x=labels,
                y=[after.get(k, 0.0) for k in metric_keys],
                marker_color=_COLORS[0],
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Impact of Query Refinement on Retrieval Quality",
        yaxis_range=[0, 1],
        **_base_layout(),
    )
    st.plotly_chart(fig, use_container_width=True)
