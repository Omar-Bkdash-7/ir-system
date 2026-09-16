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
            top_k = st.number_input(
                "Top K", 
                min_value=10, 
                max_value=1000, 
                value=100, 
                step=10, 
                key="eval_topk",
                help="Number of documents to retrieve per query"
            )
            query_limit = st.number_input(
                "Query Limit", 
                min_value=0, 
                max_value=1000, 
                value=0,
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
        
        # Query refinement toggle
        enable_refinement = st.checkbox(
            "✨ Enable Query Refinement",
            value=False,
            key="eval_refinement_toggle",
            help="Apply query refinement (high-confidence spelling correction only) during evaluation"
        )
        
        
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
                        "include_per_query": False,
                        "enable_refinement": enable_refinement
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
                        
                        # Preserve dataset selection after rerun
                        if dataset:
                            st.session_state["last_evaluated_dataset"] = dataset
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Evaluation failed: {response.error}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def _evaluation_history_panel():
    """Show evaluation history from database"""
    with st.expander("📜 Evaluation History", expanded=False):
        try:
            from api.client import get, delete
            response = get(f"{config.EVALUATION_URL}/history?limit=100")
            
            if response.ok and response.data:
                history = response.data.get("history", [])
                
                if history:
                    # Add bulk delete options
                    st.markdown("#### 🗑️ Bulk Delete Options")
                    
                    col_del1, col_del2 = st.columns(2)
                    
                    with col_del1:
                        # Delete by model
                        models_in_history = sorted(list(set(h.get("model", "") for h in history if h.get("model"))))
                        if models_in_history:
                            selected_model_to_delete = st.selectbox(
                                "Delete all results for model:",
                                ["Select..."] + models_in_history,
                                key="bulk_delete_model_select"
                            )
                            
                            if st.button("🗑️ Delete by Model", type="secondary", key="bulk_delete_model_btn", use_container_width=True):
                                if selected_model_to_delete != "Select...":
                                    with st.spinner(f"Deleting all results for {selected_model_to_delete}..."):
                                        try:
                                            del_response = delete(
                                                f"{config.EVALUATION_URL}/results?model={selected_model_to_delete}"
                                            )
                                            if del_response.ok:
                                                count = del_response.data.get("count", 0)
                                                st.success(f"✅ Deleted {count} results for {selected_model_to_delete}")
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed to delete: {del_response.error}")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                                else:
                                    st.warning("⚠️ Please select a model first")
                    
                    with col_del2:
                        # Delete by dataset
                        datasets_in_history = sorted(list(set(h.get("dataset", "") for h in history if h.get("dataset"))))
                        if datasets_in_history:
                            selected_dataset_to_delete = st.selectbox(
                                "Delete all results for dataset:",
                                ["Select..."] + datasets_in_history,
                                key="bulk_delete_dataset_select"
                            )
                            
                            if st.button("🗑️ Delete by Dataset", type="secondary", key="bulk_delete_dataset_btn", use_container_width=True):
                                if selected_dataset_to_delete != "Select...":
                                    with st.spinner(f"Deleting all results for {selected_dataset_to_delete}..."):
                                        try:
                                            del_response = delete(
                                                f"{config.EVALUATION_URL}/results?dataset={selected_dataset_to_delete}"
                                            )
                                            if del_response.ok:
                                                count = del_response.data.get("count", 0)
                                                st.success(f"✅ Deleted {count} results for {selected_dataset_to_delete}")
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed to delete: {del_response.error}")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                                else:
                                    st.warning("⚠️ Please select a dataset first")
                    
                    st.markdown("---")
                    st.markdown("#### 📋 Evaluation Records")
                    
                    # Display history as compact list
                    for idx, item in enumerate(history):
                        model_name = item.get("model", "").upper()
                        dataset_name = item.get("dataset", "")
                        date_str = item.get("created_at", "")[:19] if item.get("created_at") else ""
                        
                        # Extract refinement setting from metadata
                        metadata = item.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                        enable_refinement = metadata.get("enable_refinement", False)
                        refinement_badge = "✨ With Refinement" if enable_refinement else "❌ Without Refinement"
                        
                        # Create compact row with fixed widths
                        col_info, col_delete = st.columns([5, 1])
                        
                        with col_info:
                            st.markdown(
                                f"""
                                <div style='padding: 0.25rem 0; font-family: monospace;'>
                                    <span style='display: inline-block; width: 180px; font-weight: bold;'>{model_name}</span>
                                    <span style='display: inline-block; width: 200px;'>on <code>{dataset_name}</code></span>
                                    <span style='display: inline-block; width: 200px;'>{refinement_badge}</span>
                                    <span style='display: inline-block;'>🕒 {date_str}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col_delete:
                            if st.button("🗑️", key=f"delete_eval_{item.get('id')}", help="Delete this evaluation"):
                                with st.spinner("Deleting..."):
                                    try:
                                        del_response = delete(f"{config.EVALUATION_URL}/result/{item.get('id')}")
                                        if del_response.ok:
                                            st.success("✅ Deleted")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ {del_response.error}")
                                    except Exception as e:
                                        st.error(f"❌ {str(e)}")
                    
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
    _run_evaluation_panel()
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

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            height: 4.3rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ── Get available datasets from history ────────────────────────────────
    try:
        from api.client import get
        history_resp = get(f"{config.EVALUATION_URL}/history?limit=100")
        available_datasets = []
        
        if history_resp.ok and history_resp.data:
            history = history_resp.data.get("history", [])
            datasets_set = set(item.get("dataset") for item in history if item.get("dataset"))
            available_datasets = sorted(list(datasets_set))
    except:
        available_datasets = []

    # ── Dataset selector & source indicator ────────────────────────────────
    col_filter, col_models, col_source = st.columns([2, 1, 1])
    
    with col_filter:
        if available_datasets:
            # Check if there's a recently evaluated dataset
            default_dataset = "All Datasets"
            if "last_evaluated_dataset" in st.session_state:
                last_ds = st.session_state["last_evaluated_dataset"]
                if last_ds in available_datasets:
                    default_dataset = last_ds
            
            # Find default index
            options = ["All Datasets"] + available_datasets
            default_index = options.index(default_dataset) if default_dataset in options else 0
            
            selected_dataset = st.selectbox(
                "📂 Filter by Dataset",
                options,
                index=default_index,
                key="dataset_filter",
                help="View evaluation results for a specific dataset"
            )
        else:
            selected_dataset = "All Datasets"
            st.info("📂 Showing all available results")
        
    with col_models:
        st.metric("#### Models", len(metrics))

    with col_source:
        if data_source == "precomputed":
            st.info("📂 Pre-computed")
        else:
            st.success("🟢 Live data")


    # ── Filter metrics by dataset if selected ──────────────────────────────
    if selected_dataset and selected_dataset != "All Datasets":
        try:
            history_resp = get(f"{config.EVALUATION_URL}/history?limit=100")
            if history_resp.ok and history_resp.data:
                history = history_resp.data.get("history", [])
                
                # Filter history by dataset
                filtered_history = [h for h in history if h.get("dataset") == selected_dataset]
                
                # Get latest result per model for this dataset
                dataset_metrics = {}
                seen_models = set()
                
                for item in filtered_history:
                    model = item.get("model", "")
                    if model and model not in seen_models:
                        model_name = model.upper().replace("_", "-")
                        if model_name == "TFIDF":
                            model_name = "TF-IDF"
                        
                        dataset_metrics[model_name] = {
                            "map": item.get("map", 0),
                            "ndcg_at_10": item.get("ndcg", 0),
                            "recall": item.get("recall", 0),
                            "precision_at_10": item.get("precision", 0),
                            "num_queries": item.get("num_queries", 0)
                        }
                        seen_models.add(model)
                
                if dataset_metrics:
                    metrics = dataset_metrics
                    st.success(f"📂 Showing **{len(dataset_metrics)} models** evaluated on **{selected_dataset}**")
                else:
                    st.warning(f"No evaluation results found for dataset: {selected_dataset}")
                    return
        except Exception as e:
            st.error(f"Failed to filter by dataset: {e}")
    else:
        # Show summary for all datasets
        avg_map = sum(m.get("map", 0) for m in metrics.values()) / len(metrics) if metrics else 0
        st.info(f"📊 Showing all results | Avg MAP: **{avg_map:.4f}**")

    st.divider()
    
    # ── Model rankings ─────────────────────────────────────────────────────
    _model_comparison_panel(metrics)
    st.divider()

    # ── Model Performance Overview as Table ────────────────────────────────
    st.markdown("### 📊 Model Performance Overview")
    
    # Build table data with ALL evaluations from history
    import pandas as pd
    
    table_rows = []
    
    try:
        from api.client import get
        history_resp = get(f"{config.EVALUATION_URL}/history?limit=100")
        
        if history_resp.ok and history_resp.data:
            history = history_resp.data.get("history", [])
            
            # Filter by selected dataset if needed
            if selected_dataset and selected_dataset != "All Datasets":
                history = [h for h in history if h.get("dataset") == selected_dataset]
            
            # Build table from ALL history records
            for h in history:
                model = h.get("model", "").upper().replace("_", "-")
                if model == "TFIDF":
                    model = "TF-IDF"
                
                dataset = h.get("dataset", "")
                
                metadata = h.get("metadata", {})
                if isinstance(metadata, dict):
                    enable_refinement = metadata.get("enable_refinement", False)
                    refinement_status = "✨ With Refinement" if enable_refinement else "Without Refinement"
                else:
                    refinement_status = "Unknown"
                
                table_rows.append({
                    "Model": model,
                    "Dataset": dataset,
                    "Refinement": refinement_status,
                    "MAP": f"{h.get('map', 0):.4f}",
                    "Recall@100": f"{h.get('recall', 0):.4f}",
                    "P@10": f"{h.get('precision', 0):.4f}",
                    "nDCG@10": f"{h.get('ndcg', 0):.4f}",
                })
    except Exception as e:
        st.warning(f"Could not load evaluation data: {e}")
    
    if table_rows:
        df = pd.DataFrame(table_rows)
        
        # Sort by Model name, then by Dataset, then by Refinement (Without first, then With)
        df["sort_refinement"] = df["Refinement"].apply(
            lambda x: 0 if "Without" in x else (1 if "With" in x else 2)
        )
        df = df.sort_values(by=["Model", "Dataset", "sort_refinement"]).drop(columns=["sort_refinement"])
        df = df.reset_index(drop=True)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=None
        )
    else:
        st.info("No evaluation data available. Run evaluations first.")
    
    st.divider()
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

    # ── Refinement comparison ──────────────────────────────────────────────
    st.markdown("### 🔧 Impact of Query Refinement")
    
    # Try to get before/after data from history for current dataset
    try:
        from api.client import get
        history_resp = get(f"{config.EVALUATION_URL}/history?limit=100")
        
        if history_resp.ok and history_resp.data:
            history = history_resp.data.get("history", [])
            
            # Filter by selected dataset
            target_dataset = selected_dataset if selected_dataset != "All Datasets" else None
            
            if not target_dataset:
                st.info("📂 Please select a specific dataset from the filter above to view query refinement comparison.")
            else:
                # Filter history by selected dataset
                dataset_history = [h for h in history if h.get("dataset") == target_dataset]
                
                if not dataset_history:
                    st.info(f"No evaluation history found for dataset: {target_dataset}")
                else:
                    # Find models that have both with/without refinement evaluations
                    with_ref = {}
                    without_ref = {}
                    
                    for item in dataset_history:
                        model = item.get("model", "")
                        metadata = item.get("metadata")
                        
                        if not isinstance(metadata, dict):
                            metadata = {}
                        
                        has_refinement = metadata.get("enable_refinement", False)
                        
                        if has_refinement and model not in with_ref:
                            with_ref[model] = {
                                "map": item.get("map", 0),
                                "ndcg_at_10": item.get("ndcg", 0),
                                "recall": item.get("recall", 0),
                                "precision_at_10": item.get("precision", 0)
                            }
                        elif not has_refinement and model not in without_ref:
                            without_ref[model] = {
                                "map": item.get("map", 0),
                                "ndcg_at_10": item.get("ndcg", 0),
                                "recall": item.get("recall", 0),
                                "precision_at_10": item.get("precision", 0)
                            }
                    
                    # Find models with both evaluations
                    available_models = sorted(list(set(with_ref.keys()) & set(without_ref.keys())))
                    
                    if not available_models:
                        st.info("📊 No models with both refinement and non-refinement evaluations found.")
                        st.caption("💡 Tip: Run evaluation with and without '✨ Enable Query Refinement' for the same model.")
                    else:
                        # Model selector
                        selected_model = st.selectbox(
                            "🤖 Select Model for Comparison",
                            available_models,
                            format_func=lambda x: x.upper().replace("_", " ").title(),
                            key="refinement_model_selector"
                        )
                        
                        if selected_model:
                            before = without_ref[selected_model]
                            after = with_ref[selected_model]
                            
                            st.markdown(f"#### 💡 Comparison for {selected_model.upper()} on {target_dataset} dataset")
                            
                            # Tabs for different visualizations
                            ref_tab1, ref_tab2, ref_tab3 = st.tabs(["📊 Bar Chart", "🕸️ Radar", "📉 Precision@K"])
                            
                            with ref_tab1:
                                render_refinement_comparison_chart(before, after)
                            
                            with ref_tab2:
                                comparison_metrics = {
                                    "Before Refinement": before,
                                    "After Refinement": after
                                }
                                render_radar_chart(comparison_metrics)
                            
                            with ref_tab3:
                                p10_before = before.get("precision_at_10", 0.0)
                                p10_after = after.get("precision_at_10", 0.0)
                                
                                pk_comparison: Dict[str, Dict[int, float]] = {}
                                
                                if p10_before > 0:
                                    pk_comparison["Before Refinement"] = {
                                        1: min(p10_before * 1.3, 1.0),
                                        3: min(p10_before * 1.2, 1.0),
                                        5: min(p10_before * 1.1, 1.0),
                                        10: p10_before,
                                        20: p10_before * 0.85,
                                        50: p10_before * 0.65,
                                        100: p10_before * 0.5,
                                    }
                                
                                if p10_after > 0:
                                    pk_comparison["After Refinement"] = {
                                        1: min(p10_after * 1.3, 1.0),
                                        3: min(p10_after * 1.2, 1.0),
                                        5: min(p10_after * 1.1, 1.0),
                                        10: p10_after,
                                        20: p10_after * 0.85,
                                        50: p10_after * 0.65,
                                        100: p10_after * 0.5,
                                    }
                                
                                if pk_comparison:
                                    render_precision_at_k_curve(pk_comparison)
                                    st.caption("P@K values for K≠10 are extrapolated from P@10.")
                            
                            st.markdown("---")
                            
                            # Show improvement stats
                            st.markdown("##### 📊 Performance Improvement")
                            
                            map_improvement = ((after["map"] - before["map"]) / before["map"] * 100) if before["map"] > 0 else 0
                            ndcg_improvement = ((after["ndcg_at_10"] - before["ndcg_at_10"]) / before["ndcg_at_10"] * 100) if before["ndcg_at_10"] > 0 else 0
                            p10_improvement = ((after["precision_at_10"] - before["precision_at_10"]) / before["precision_at_10"] * 100) if before["precision_at_10"] > 0 else 0
                            recall_improvement = ((after["recall"] - before["recall"]) / before["recall"] * 100) if before["recall"] > 0 else 0
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(
                                    f"""
                                    <div style='text-align: center; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 0.5rem;'>
                                        <div style='font-size: 0.875rem; margin-bottom: 0.5rem;'>📊 MAP Improvement</div>
                                        <div style='font-size: 1.5rem; font-weight: bold; color: {"#28a745" if map_improvement > 0 else "#dc3545"};'>{map_improvement:+.2f}%</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col2:
                                st.markdown(
                                    f"""
                                    <div style='text-align: center; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 0.5rem;'>
                                        <div style='font-size: 0.875rem; margin-bottom: 0.5rem;'>⭐ nDCG@10 Improvement</div>
                                        <div style='font-size: 1.5rem; font-weight: bold; color: {"#28a745" if ndcg_improvement > 0 else "#dc3545"};'>{ndcg_improvement:+.2f}%</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col3:
                                st.markdown(
                                    f"""
                                    <div style='text-align: center; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 0.5rem;'>
                                        <div style='font-size: 0.875rem; margin-bottom: 0.5rem;'>🔍 P@10 Improvement</div>
                                        <div style='font-size: 1.5rem; font-weight: bold; color: {"#28a745" if p10_improvement > 0 else "#dc3545"};'>{p10_improvement:+.2f}%</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col4:
                                st.markdown(
                                    f"""
                                    <div style='text-align: center; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 0.5rem;'>
                                        <div style='font-size: 0.875rem; margin-bottom: 0.5rem;'>🎯 Recall Improvement</div>
                                        <div style='font-size: 1.5rem; font-weight: bold; color: {"#28a745" if recall_improvement > 0 else "#dc3545"};'>{recall_improvement:+.2f}%</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
        else:
            st.info("No evaluation history available.")
    except Exception as e:
        st.error(f"Error loading refinement comparison: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

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
