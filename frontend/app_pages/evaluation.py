"""
Evaluation page — IR metrics dashboard with charts.
Falls back to pre-computed JSON files when the evaluation service is a stub.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import streamlit as st

import api.evaluation as eval_api
import config
from api.client import post
from components.metrics_chart import (
    render_metrics_bar_chart,
    render_precision_at_k_curve,
    render_radar_chart,
    render_refinement_comparison_chart,
)
from utils.dataset_utils import get_available_datasets, get_dataset_display_name

# Paths to pre-computed evaluation JSON files
_EVAL_DIR = Path("/app/data/evaluation")
_LOCAL_EVAL_DIR = Path(__file__).parent.parent.parent / "data" / "evaluation"

_PRECOMPUTED_FILES = {
    "TF-IDF":    ("tfidf_metrics.json",     "tfidf_results.json"),
    "Embedding": ("embedding_metrics.json",  "embedding_results.json"),
}

_METRIC_KEY_MAP = {
    "MAP": "map", "map": "map",
    "Recall": "recall", "recall": "recall",
    "P@10": "precision_at_10", "precision_at_10": "precision_at_10",
    "nDCG@10": "ndcg_at_10", "ndcg_at_10": "ndcg_at_10",
}


def _load_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _load_precomputed() -> Dict[str, dict]:
    """Search container path and local dev path for pre-computed metric files."""
    metrics: Dict[str, dict] = {}
    for candidate_dir in (_EVAL_DIR, _LOCAL_EVAL_DIR):
        for model_label, (metrics_file, _) in _PRECOMPUTED_FILES.items():
            if model_label in metrics:
                continue
            data = _load_json(candidate_dir / metrics_file)
            if data:
                metrics[model_label] = data
    return metrics


def _normalize_metrics(raw: Dict[str, dict]) -> Dict[str, dict]:
    """Normalize heterogeneous key names into a canonical schema."""
    out: Dict[str, dict] = {}
    for model, vals in raw.items():
        out[model] = {_METRIC_KEY_MAP.get(k, k): v for k, v in vals.items()}
    return out


def _metric_table(metrics: Dict[str, dict]) -> None:
    import pandas as pd
    rows = []
    for model, vals in metrics.items():
        rows.append({
            "Model":      model,
            "MAP":        f"{vals.get('map', 0):.4f}",
            "Recall@100": f"{vals.get('recall', 0):.4f}",
            "P@10":       f"{vals.get('precision_at_10', 0):.4f}",
            "nDCG@10":    f"{vals.get('ndcg_at_10', 0):.4f}",
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("Model"),
            use_container_width=True,
        )


def _run_evaluation_panel():
    """Interactive panel to run new evaluations"""
    with st.expander("🚀 Run New Evaluation", expanded=False):
        st.markdown("Run evaluation on demand to get fresh metrics.")
        
        available_datasets = get_available_datasets()
        
        if not available_datasets:
            st.warning("⚠️ No datasets with all indices (TF-IDF, BM25, Embedding) are available. Please build indices first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox(
                "Model",
                ["tfidf", "embedding", "bm25", "hybrid_serial", "hybrid_parallel"],
                format_func=lambda x: x.upper().replace("_", " ").title(),
                key="eval_model_select"
            )
            
            dataset = st.selectbox(
                "Dataset",
                available_datasets,
                format_func=get_dataset_display_name,
                help="Only datasets with all indices (TF-IDF, BM25, Embedding) are shown",
                key="eval_dataset_select"
            )
        
        with col2:
            top_k = st.number_input("Top K", min_value=10, max_value=1000, value=100, step=10, key="eval_topk")
            query_limit = st.number_input(
                "Query Limit (0=all)", 
                min_value=0, 
                max_value=1000, 
                value=100,
                help="Limit queries for faster testing. 0 means all queries.",
                key="eval_query_limit"
            )
        
        # Hybrid configuration
        hybrid_models = None
        fusion_method = "rrf"
        
        if model in ["hybrid_serial", "hybrid_parallel"]:
            st.markdown("##### Hybrid Configuration")
            
            if model == "hybrid_serial":
                col_a, col_b = st.columns(2)
                with col_a:
                    first_model = st.selectbox(
                        "First Model",
                        ["tfidf", "bm25", "embedding"],
                        index=1,
                        key="eval_first_model"
                    )
                with col_b:
                    second_model = st.selectbox(
                        "Second Model",
                        ["tfidf", "bm25", "embedding"],
                        index=2,
                        key="eval_second_model"
                    )
                hybrid_models = [first_model, second_model]
            
            else:  # hybrid_parallel
                st.markdown(
                    """
                    <style>
                    span[data-baseweb="tag"] span {
                        color: #0f172a !important;
                        font-weight: 600 !important;
                    }

                    span[data-baseweb="tag"] svg {
                        fill: #0f172a !important;
                        color: #0f172a !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                hybrid_models = st.multiselect(
                    "Models (select 2+)",
                    ["tfidf", "bm25", "embedding"],
                    default=["tfidf", "bm25"],
                    key="eval_parallel_models"
                )
                
                fusion_method = st.selectbox(
                    "Fusion Method",
                    ["rrf", "combsum", "combmnz", "weighted"],
                    key="eval_fusion_method"
                )
                
                if len(hybrid_models) < 2:
                    st.warning("⚠️ Select at least 2 models for hybrid parallel")
        
        if st.button("▶️ Run Evaluation", type="primary", use_container_width=True, key="eval_run_btn"):
            # Validation
            if model in ["hybrid_serial", "hybrid_parallel"]:
                if not hybrid_models or len(hybrid_models) < 2:
                    st.error("❌ Hybrid models require at least 2 models")
                    return
            
            with st.spinner(f"Evaluating {model.upper()} on {dataset}..."):
                try:
                    payload = {
                        "dataset_name": dataset,
                        "model": model,
                        "top_k": top_k,
                        "query_limit": query_limit if query_limit > 0 else None,
                        "include_per_query": False
                    }
                    
                    # Add hybrid-specific parameters
                    if model in ["hybrid_serial", "hybrid_parallel"]:
                        payload["hybrid_models"] = hybrid_models
                        if model == "hybrid_parallel":
                            payload["fusion_method"] = fusion_method
                    
                    # Evaluation can take long time, especially for hybrid models
                    # Use extended timeout (100 minutes)
                    response = post(f"{config.EVALUATION_URL}/evaluate", json=payload, timeout=6000)
                    
                    if response.ok:
                        st.success(f"✅ Evaluation completed!")
                        result = response.data
                        
                        st.markdown("#### Results:")
                        metrics = result.get("metrics", {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MAP", f"{metrics.get('map', 0):.4f}")
                        col2.metric("nDCG", f"{metrics.get('ndcg', 0):.4f}")
                        col3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                        col4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                        
                        st.info(f"📊 Evaluated on {metrics.get('num_queries', 0)} queries")
                        st.balloons()
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Evaluation failed: {response.error}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def _evaluation_history_panel():
    """Show evaluation history from database"""
    with st.expander("📜 Evaluation History", expanded=False):
        try:
            from api.client import get
            response = get(f"{config.EVALUATION_URL}/history?limit=10")
            
            if response.ok and response.data:
                history = response.data.get("history", [])
                
                if history:
                    import pandas as pd
                    
                    df_data = []
                    for item in history:
                        df_data.append({
                            "Model": item.get("model", "").upper(),
                            "Dataset": item.get("dataset", ""),
                            "MAP": f"{item.get('map', 0):.4f}",
                            "nDCG": f"{item.get('ndcg', 0):.4f}",
                            "Recall": f"{item.get('recall', 0):.4f}",
                            "Queries": item.get("num_queries", 0),
                            "Date": item.get("created_at", "")[:19] if item.get("created_at") else ""
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No evaluation history found. Run some evaluations first!")
            else:
                st.info("Evaluation history not available.")
                
        except Exception as e:
            st.warning(f"Could not load history: {str(e)}")


def _model_comparison_panel(metrics: Dict[str, dict]):
    """Compare models side by side"""
    st.markdown("### 🏆 Model Rankings")
    
    # Rank models by each metric
    metric_keys = ["map", "ndcg_at_10", "recall", "precision_at_10"]
    metric_labels = ["MAP", "nDCG@10", "Recall@100", "P@10"]
    
    cols = st.columns(4)
    for col, key, label in zip(cols, metric_keys, metric_labels):
        with col:
            sorted_models = sorted(
                metrics.items(), 
                key=lambda x: x[1].get(key, 0), 
                reverse=True
            )
            
            st.markdown(f"**{label}**")
            for i, (model, vals) in enumerate(sorted_models[:3], 1):
                medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                value = vals.get(key, 0)
                st.markdown(f"{medal} {model}: `{value:.4f}`")


def render() -> None:
    st.markdown("## 📈 Evaluation Dashboard")
    st.markdown("Comprehensive IR system performance metrics and model comparison")
    
    # ── Interactive controls ───────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        _run_evaluation_panel()
    with col2:
        _evaluation_history_panel()
    
    st.divider()

    # ── Data source ────────────────────────────────────────────────────────
    live_resp = eval_api.get_metrics()
    data_source = "live"
    raw_metrics: Dict[str, dict] = {}

    if live_resp.ok and isinstance(live_resp.data, dict) and live_resp.data:
        raw_metrics = live_resp.data
    else:
        raw_metrics = _load_precomputed()
        data_source = "precomputed"

    if not raw_metrics:
        st.warning(
            "⚠️ No evaluation data available. Use the **Run New Evaluation** panel above to generate metrics."
        )
        return

    metrics = _normalize_metrics(raw_metrics)

    # Data source indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if data_source == "precomputed":
            st.info("📂 Showing pre-computed results from disk")
        else:
            st.success("🟢 Live data from evaluation service")
    with col2:
        st.metric("Models", len(metrics))
    with col3:
        avg_map = sum(m.get("map", 0) for m in metrics.values()) / len(metrics) if metrics else 0
        st.metric("Avg MAP", f"{avg_map:.4f}")

    st.divider()
    
    # ── Model rankings ─────────────────────────────────────────────────────
    _model_comparison_panel(metrics)
    st.divider()

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("### 📊 Detailed Metrics")
    _metric_table(metrics)
    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────
    st.markdown("### 📈 Visual Analysis")
    st.markdown(
    """
    <style>
    div[data-baseweb="tab-list"] {
        gap: 16px !important;
    }
    button[data-baseweb="tab"] {
        padding: 8px 18px !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🕸️ Radar", "📉 Precision@K"])
    with tab1:
        render_metrics_bar_chart(metrics)
    with tab2:
        render_radar_chart(metrics)
    with tab3:
        if metrics:
            pk_data: Dict[str, Dict[int, float]] = {}
            for model, vals in metrics.items():
                p10 = vals.get("precision_at_10", 0.0)
                if p10 > 0:
                    pk_data[model] = {
                        1: min(p10 * 1.3, 1.0),
                        3: min(p10 * 1.2, 1.0),
                        5: min(p10 * 1.1, 1.0),
                        10: p10,
                        20: p10 * 0.85,
                        50: p10 * 0.65,
                        100: p10 * 0.5,
                    }
            
            if pk_data:
                render_precision_at_k_curve(pk_data)
                st.caption("P@K values for K≠10 are extrapolated from P@10.")
            else:
                st.info("No precision data available for visualization.")
        else:
            st.info("No metrics available for Precision@K curve.")

    st.divider()

    # ── Per-model detail cards ─────────────────────────────────────────────
    st.markdown("### 🎯 Per-Model Performance")
    
    metric_keys = [
        ("map", "MAP", "📊"), 
        ("recall", "Recall@100", "🎯"),
        ("precision_at_10", "P@10", "🔍"), 
        ("ndcg_at_10", "nDCG@10", "⭐"),
    ]
    
    # Show models in cards
    num_models = len(metrics)
    cols_per_row = min(num_models, 3)
    
    model_items = list(metrics.items())
    for i in range(0, num_models, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < num_models:
                model_name, vals = model_items[i + j]
                with col:
                    with st.container():
                        st.markdown(f"#### {model_name}")
                        
                        for key, label, icon in metric_keys:
                            value = vals.get(key, 0.0)
                            # Color code based on performance
                            if value >= 0.7:
                                color = "🟢"
                            elif value >= 0.5:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            st.metric(
                                f"{icon} {label}", 
                                f"{value:.4f}",
                                delta=None
                            )
                        
                        # Overall score
                        overall = sum(vals.get(k, 0) for k, _, _ in metric_keys) / len(metric_keys)
                        st.metric("📈 Overall", f"{overall:.4f}")

    st.divider()

    # ── Refinement comparison ──────────────────────────────────────────────
    st.markdown("### 🔧 Impact of Query Refinement")
    ref_resp = eval_api.get_refinement_comparison()
    if ref_resp.ok and isinstance(ref_resp.data, dict):
        before = ref_resp.data.get("before", {})
        after  = ref_resp.data.get("after", {})
        if before and after:
            render_refinement_comparison_chart(before, after)
        else:
            st.info("Refinement comparison data not available from the evaluation service.")
    else:
        first_model_vals = next(iter(metrics.values()), {})
        if first_model_vals:
            synthetic_before = {k: v * 0.88 for k, v in first_model_vals.items()
                                if isinstance(v, float)}
            render_refinement_comparison_chart(synthetic_before, first_model_vals)
            st.caption(
                "⚠️ Synthetic data — before-refinement values estimated as 88% of base metrics."
            )

    st.divider()

    # ── Additional insights ────────────────────────────────────────────────
    st.markdown("### 💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Best model per metric
        st.markdown("#### 🏆 Best Performers")
        for key, label, _ in [
            ("map", "MAP", "📊"),
            ("ndcg_at_10", "nDCG@10", "⭐"),
            ("recall", "Recall", "🎯"),
            ("precision_at_10", "P@10", "🔍")
        ]:
            best_model = max(metrics.items(), key=lambda x: x[1].get(key, 0))
            st.markdown(f"**{label}**: {best_model[0]} `({best_model[1].get(key, 0):.4f})`")
    
    with col2:
        # Performance distribution
        st.markdown("#### 📊 Performance Overview")
        all_maps = [v.get("map", 0) for v in metrics.values()]
        if all_maps:
            avg_map = sum(all_maps) / len(all_maps)
            max_map = max(all_maps)
            min_map = min(all_maps)
            
            st.markdown(f"**Average MAP**: `{avg_map:.4f}`")
            st.markdown(f"**Best MAP**: `{max_map:.4f}`")
            st.markdown(f"**Worst MAP**: `{min_map:.4f}`")
            st.markdown(f"**Range**: `{max_map - min_map:.4f}`")

    st.divider()

    # ── Export and raw data ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Metrics as JSON", use_container_width=True):
            json_str = json.dumps(raw_metrics, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        with st.expander("📄 View Raw JSON Data"):
            st.json(raw_metrics)
