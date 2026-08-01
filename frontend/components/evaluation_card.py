"""
Evaluation card components for better visualization.
"""
from __future__ import annotations
from typing import Dict
import streamlit as st


def render_model_card(model_name: str, metrics: Dict[str, float], rank: int = None):
    """
    Render a beautiful card for a single model's metrics.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metric values
        rank: Optional ranking position
    """
    # Card styling
    card_css = """
    <style>
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .model-card-header {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .metric-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        color: white;
    }
    .metric-label {
        font-size: 12px;
        opacity: 0.8;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """
    
    with st.container():
        # Rank badge
        if rank:
            rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            st.markdown(f"### {rank_emoji} {model_name}")
        else:
            st.markdown(f"### {model_name}")
        
        # Metrics in columns
        metric_defs = [
            ("map", "MAP", "📊"),
            ("ndcg_at_10", "nDCG@10", "⭐"),
            ("recall", "Recall", "🎯"),
            ("precision_at_10", "P@10", "🔍")
        ]
        
        cols = st.columns(2)
        for i, (key, label, icon) in enumerate(metric_defs):
            with cols[i % 2]:
                value = metrics.get(key, 0.0)
                
                # Performance indicator
                if value >= 0.7:
                    delta_color = "normal"
                    perf = "Excellent"
                elif value >= 0.5:
                    delta_color = "off"
                    perf = "Good"
                else:
                    delta_color = "inverse"
                    perf = "Needs Work"
                
                st.metric(
                    label=f"{icon} {label}",
                    value=f"{value:.4f}",
                    help=f"Performance: {perf}"
                )


def render_comparison_badge(value: float, threshold_high: float = 0.7, threshold_low: float = 0.5) -> str:
    """
    Return a colored badge based on metric value.
    
    Args:
        value: Metric value (0-1)
        threshold_high: Threshold for excellent performance
        threshold_low: Threshold for good performance
        
    Returns:
        HTML badge string
    """
    if value >= threshold_high:
        color = "#10b981"  # green
        label = "Excellent"
    elif value >= threshold_low:
        color = "#f59e0b"  # amber
        label = "Good"
    else:
        color = "#ef4444"  # red
        label = "Poor"
    
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{label}</span>'


def render_metric_gauge(value: float, label: str, max_value: float = 1.0):
    """
    Render a gauge/progress bar for a metric.
    
    Args:
        value: Current value
        label: Metric label
        max_value: Maximum value for the gauge
    """
    percentage = int((value / max_value) * 100)
    
    # Color based on performance
    if percentage >= 70:
        color = "#10b981"
    elif percentage >= 50:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">{label}</span>
            <span>{value:.4f}</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 10px; height: 20px; overflow: hidden;">
            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_leaderboard(metrics: Dict[str, Dict[str, float]], metric_key: str = "map"):
    """
    Render a leaderboard sorted by a specific metric.
    
    Args:
        metrics: Dictionary of model metrics
        metric_key: Key to sort by
    """
    st.markdown("#### 🏆 Leaderboard")
    
    sorted_models = sorted(
        metrics.items(),
        key=lambda x: x[1].get(metric_key, 0),
        reverse=True
    )
    
    for i, (model, vals) in enumerate(sorted_models, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
        value = vals.get(metric_key, 0)
        
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.markdown(f"**{medal}**")
        with col2:
            st.markdown(f"**{model}**")
        with col3:
            st.markdown(f"`{value:.4f}`")
