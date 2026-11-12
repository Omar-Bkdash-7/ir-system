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

from obs import failure_hint, get_logger, log_failure

logger = get_logger(__name__)

import api.evaluation as eval_api
from components.metrics_chart import (
    render_metrics_bar_chart,
    render_precision_at_k_curve,
    render_radar_chart,
    render_refinement_comparison_chart,
)
from ui_style import inject_style
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


def _render_evaluation_inputs(available_datasets: list[str]) -> tuple[str, str, int, int]:
    """Model, dataset, top_k and the query limit."""
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
            value=100,
            help="Limit queries for faster testing. 0 means all queries.",
            key="eval_query_limit"
        )

    return model, dataset, top_k, query_limit


def _render_hybrid_configuration(model: str) -> tuple[Optional[list], str]:
    """Which models feed a hybrid run, and how parallel results are fused."""
    hybrid_models = None
    fusion_method = "rrf"

    if model not in ["hybrid_serial", "hybrid_parallel"]:
        return hybrid_models, fusion_method

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
        inject_style("dataframe_tags")
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

    return hybrid_models, fusion_method


def _render_evaluation_refinement_toggle() -> bool:
    """Refinement is spelling-only during evaluation, and says so."""
    enable_refinement = st.checkbox(
        "✨ Enable Query Refinement",
        value=False,
        key="eval_refinement_toggle",
        help="Apply intelligent query refinement (spelling correction only in evaluation mode)"
    )

    if enable_refinement:
        st.info(
            "ℹ️ **Evaluation Mode**: Only high-confidence spelling correction is applied. "
            "Expansion and history features are disabled for deterministic results."
        )

    return enable_refinement


def _render_evaluation_result(result: dict, dataset: str) -> None:
    """Metrics of a finished run, then rerun so the tables pick it up."""
    st.success("✅ Evaluation completed!")

    st.markdown("#### Results:")
    metrics = result.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAP", f"{metrics.get('map', 0):.4f}")
    col2.metric("nDCG", f"{metrics.get('ndcg', 0):.4f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    col4.metric("Precision", f"{metrics.get('precision', 0):.4f}")

    st.info(f"📊 Evaluated on {metrics.get('num_queries', 0)} queries")
    st.balloons()

    if dataset:
        st.session_state["last_evaluated_dataset"] = dataset

    st.rerun()


def _run_evaluation_now(model, dataset, top_k, query_limit,
                        hybrid_models, fusion_method, enable_refinement) -> None:
    """Validate the selection, call the service, show the metrics."""
    if model in ["hybrid_serial", "hybrid_parallel"]:
        if not hybrid_models or len(hybrid_models) < 2:
            st.error("❌ Hybrid models require at least 2 models")
            return

    with st.spinner(f"Evaluating {model.upper()} on {dataset}..."):
        payload = {
            "dataset_name": dataset,
            "model": model,
            "top_k": top_k,
            "query_limit": query_limit if query_limit > 0 else None,
            "include_per_query": False,
            "enable_refinement": enable_refinement
        }

        if model in ["hybrid_serial", "hybrid_parallel"]:
            payload["hybrid_models"] = hybrid_models
            if model == "hybrid_parallel":
                payload["fusion_method"] = fusion_method

        # Evaluation can take a long time, especially for hybrid models:
        # the client timeout for this call is raised by the API wrapper.
        response = eval_api.run_evaluation(payload)

    if response.ok:
        _render_evaluation_result(response.data, dataset)
    else:
        log_failure(logger, "run_evaluation", response)
        st.error(f"❌ Evaluation failed: {failure_hint(response)}")


def _run_evaluation_panel():
    """Interactive panel to run new evaluations"""
    with st.expander("🚀 Run New Evaluation", expanded=False):
        st.markdown("Run evaluation on demand to get fresh metrics.")

        available_datasets = get_available_datasets()

        if not available_datasets:
            st.warning("⚠️ No datasets with all indices (TF-IDF, BM25, Embedding) are available. Please build indices first.")
            return

        model, dataset, top_k, query_limit = _render_evaluation_inputs(available_datasets)
        hybrid_models, fusion_method = _render_hybrid_configuration(model)
        enable_refinement = _render_evaluation_refinement_toggle()

        if st.button("▶️ Run Evaluation", type="primary", use_container_width=True, key="eval_run_btn"):
            _run_evaluation_now(model, dataset, top_k, query_limit,
                                hybrid_models, fusion_method, enable_refinement)


def _bulk_delete(field: str, value: str) -> None:
    """Delete every stored result matching one model or one dataset."""
    if value == "Select...":
        st.warning(f"⚠️ Please select a {field} first")
        return

    with st.spinner(f"Deleting all results for {value}..."):
        del_response = eval_api.delete_results(**{field: value})

    if del_response.ok:
        count = del_response.data.get("count", 0)
        st.success(f"✅ Deleted {count} results for {value}")
        st.rerun()
    else:
        log_failure(logger, f"delete_by_{field}", del_response)
        st.error(f"❌ Failed to delete: {failure_hint(del_response)}")


def _render_bulk_delete_controls(history: list[dict]) -> None:
    """Delete everything for one model, or everything for one dataset."""
    st.markdown("#### 🗑️ Bulk Delete Options")

    col_del1, col_del2 = st.columns(2)

    with col_del1:
        models_in_history = sorted(list(set(h.get("model", "") for h in history if h.get("model"))))
        if models_in_history:
            selected_model_to_delete = st.selectbox(
                "Delete all results for model:",
                ["Select..."] + models_in_history,
                key="bulk_delete_model_select"
            )

            if st.button("🗑️ Delete by Model", type="secondary", key="bulk_delete_model_btn", use_container_width=True):
                _bulk_delete("model", selected_model_to_delete)

    with col_del2:
        datasets_in_history = sorted(list(set(h.get("dataset", "") for h in history if h.get("dataset"))))
        if datasets_in_history:
            selected_dataset_to_delete = st.selectbox(
                "Delete all results for dataset:",
                ["Select..."] + datasets_in_history,
                key="bulk_delete_dataset_select"
            )

            if st.button("🗑️ Delete by Dataset", type="secondary", key="bulk_delete_dataset_btn", use_container_width=True):
                _bulk_delete("dataset", selected_dataset_to_delete)


def _delete_history_record(item: dict) -> None:
    """Remove one stored evaluation."""
    with st.spinner("Deleting..."):
        del_response = eval_api.delete_result(item.get('id'))

    if del_response.ok:
        st.success("✅ Deleted")
        st.rerun()
    else:
        log_failure(logger, "delete_result", del_response)
        st.error(f"❌ {failure_hint(del_response)}")


def _render_evaluation_records(history: list[dict]) -> None:
    st.markdown("---")
    st.markdown("#### 📋 Evaluation Records")

    for item in history:
        model_name = item.get("model", "").upper()
        dataset_name = item.get("dataset", "")
        date_str = item.get("created_at", "")[:19] if item.get("created_at") else ""

        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        enable_refinement = metadata.get("enable_refinement", False)
        refinement_badge = "✨ With Refinement" if enable_refinement else "❌ Without Refinement"

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
                _delete_history_record(item)


def _evaluation_history_panel():
    """Show evaluation history from database"""
    with st.expander("📜 Evaluation History", expanded=False):
        response = eval_api.get_history()

        if not (response.ok and response.data):
            log_failure(logger, "evaluation_history", response)
            st.info("Evaluation history not available.")
            return

        history = response.data.get("history", [])

        if not history:
            st.info("No evaluation history found. Run some evaluations first!")
            return

        _render_bulk_delete_controls(history)
        _render_evaluation_records(history)


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

    _render_controls()

    raw_metrics, data_source = _load_metrics()
    if not raw_metrics:
        st.warning(
            "⚠️ No evaluation data available. Use the **Run New Evaluation** panel above to generate metrics."
        )
        return

    metrics = _normalize_metrics(raw_metrics)
    inject_style("metric_row")

    selected_dataset = _render_dataset_filter(metrics, data_source)
    filtered = _metrics_for_dataset(metrics, selected_dataset)
    if filtered is None:
        return
    metrics = filtered

    st.divider()
    _render_model_rankings(metrics)

    st.divider()
    _render_history_table(selected_dataset)

    st.divider()
    _render_visual_analysis(metrics)

    st.divider()
    _render_refinement_comparison(selected_dataset)

    st.divider()
    _render_insights(metrics)

    st.divider()
    _render_export(raw_metrics)


def _render_controls() -> None:
    """Run-evaluation and history panels, closed by the header divider."""
    # ── Interactive controls ───────────────────────────────────────────────
    _run_evaluation_panel()
    _evaluation_history_panel()

    st.divider()


def _load_metrics() -> tuple[Dict[str, dict], str]:
    """Live metrics from the service, or the bundled pre-computed snapshot."""
    # ── Data source ────────────────────────────────────────────────────────
    live_resp = eval_api.get_metrics()
    data_source = "live"
    raw_metrics: Dict[str, dict] = {}

    if live_resp.ok and isinstance(live_resp.data, dict) and live_resp.data:
        raw_metrics = live_resp.data
    else:
        raw_metrics = _load_precomputed()
        data_source = "precomputed"
    return raw_metrics, data_source


def _render_dataset_filter(metrics: Dict[str, dict], data_source: str) -> str:
    """Dataset picker plus the model count and the data-source badge."""
    # ── Get available datasets from history ────────────────────────────────
    try:
        history_resp = eval_api.get_history()
        available_datasets = []
        
        if history_resp.ok and history_resp.data:
            history = history_resp.data.get("history", [])
            datasets_set = set(item.get("dataset") for item in history if item.get("dataset"))
            available_datasets = sorted(list(datasets_set))
    except Exception:  # the history filter is optional; the page works without it
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
    return selected_dataset


def _metrics_for_dataset(metrics: Dict[str, dict], selected_dataset: str) -> Optional[Dict[str, dict]]:
    """Metrics of the selected dataset (latest run per model), or None when that dataset has no runs."""
    # ── Filter metrics by dataset if selected ──────────────────────────────
    if selected_dataset and selected_dataset != "All Datasets":
        try:
            history_resp = eval_api.get_history()
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
                    return None
        except Exception as e:
            st.error(f"Failed to filter by dataset: {e}")
    else:
        # Show summary for all datasets
        avg_map = sum(m.get("map", 0) for m in metrics.values()) / len(metrics) if metrics else 0
        st.info(f"📊 Showing all results | Avg MAP: **{avg_map:.4f}**")
    return metrics


def _render_model_rankings(metrics: Dict[str, dict]) -> None:
    """Leaderboard across the visible metrics."""
    _model_comparison_panel(metrics)


def _render_history_table(selected_dataset: str) -> None:
    """Full table of stored runs: model, dataset, refinement and metrics."""
    # ── Model Performance Overview as Table ────────────────────────────────
    st.markdown("### 📊 Model Performance Overview")
    
    # Build table data with ALL evaluations from history
    import pandas as pd
    
    table_rows = []
    
    try:
        history_resp = eval_api.get_history()
        
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
        
        # Sort by Model name first, then by Refinement (Without first, then With), then by Dataset
        df["sort_refinement"] = df["Refinement"].apply(
            lambda x: 0 if "Without" in x else (1 if "With" in x else 2)
        )
        df = df.sort_values(by=["Model", "sort_refinement", "Dataset"]).drop(columns=["sort_refinement"])
        df = df.reset_index(drop=True)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=None
        )
    else:
        st.info("No evaluation data available. Run evaluations first.")


def _extrapolate_precision_at_k(p10: float) -> Dict[int, float]:
    """P@K curve from a single P@10 value (the service only stores P@10)."""
    return {
        1: min(p10 * 1.3, 1.0),
        3: min(p10 * 1.2, 1.0),
        5: min(p10 * 1.1, 1.0),
        10: p10,
        20: p10 * 0.85,
        50: p10 * 0.65,
        100: p10 * 0.5,
    }


def _render_visual_analysis(metrics: Dict[str, dict]) -> None:
    """Bar chart, radar and Precision@K tabs."""
    st.markdown("### 📈 Visual Analysis")
    inject_style("tabs")
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
                    pk_data[model] = _extrapolate_precision_at_k(p10)
            
            if pk_data:
                render_precision_at_k_curve(pk_data)
                st.caption("P@K values for K≠10 are extrapolated from P@10.")
            else:
                st.info("No precision data available for visualization.")
        else:
            st.info("No metrics available for Precision@K curve.")


def _improvement_card(label: str, value: float) -> str:
    """Stat card for a percentage change; green when it improved, red otherwise."""
    color = "#28a745" if value > 0 else "#dc3545"
    return (
        "<div style='text-align: center; padding: 1rem; border: 1px solid #e0e0e0; "
        "border-radius: 0.5rem;'>"
        f"<div style='font-size: 0.875rem; margin-bottom: 0.5rem;'>{label}</div>"
        f"<div style='font-size: 1.5rem; font-weight: bold; color: {color};'>{value:+.2f}%</div>"
        "</div>"
    )


def _collect_refinement_pairs(dataset_history: list[dict]) -> tuple[dict, dict]:
    """Split stored results into with-refinement and without-refinement metrics."""
    with_ref: Dict[str, dict] = {}
    without_ref: Dict[str, dict] = {}

    for item in dataset_history:
        model = item.get("model", "")
        metadata = item.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        metrics = {
            "map": item.get("map", 0),
            "ndcg_at_10": item.get("ndcg", 0),
            "recall": item.get("recall", 0),
            "precision_at_10": item.get("precision", 0),
        }

        if metadata.get("enable_refinement", False) and model not in with_ref:
            with_ref[model] = metrics
        elif not metadata.get("enable_refinement", False) and model not in without_ref:
            without_ref[model] = metrics

    return with_ref, without_ref


def _render_refinement_tabs(before: Dict[str, float], after: Dict[str, float]) -> None:
    """Three views of the same before/after pair."""
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
            pk_comparison["Before Refinement"] = _extrapolate_precision_at_k(p10_before)

        if p10_after > 0:
            pk_comparison["After Refinement"] = _extrapolate_precision_at_k(p10_after)

        if pk_comparison:
            render_precision_at_k_curve(pk_comparison)
            st.caption("P@K values for K≠10 are extrapolated from P@10.")


def _improvement_percent(before: float, after: float) -> float:
    """Relative change, or 0 when there is no baseline to compare against."""
    return ((after - before) / before * 100) if before > 0 else 0


def _render_improvement_cards(before: Dict[str, float], after: Dict[str, float]) -> None:
    """One card per metric showing the relative gain."""
    st.markdown("---")
    st.markdown("##### 📊 Performance Improvement")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            _improvement_card("📊 MAP Improvement",
                              _improvement_percent(before["map"], after["map"])),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            _improvement_card("⭐ nDCG@10 Improvement",
                              _improvement_percent(before["ndcg_at_10"], after["ndcg_at_10"])),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            _improvement_card("🔍 P@10 Improvement",
                              _improvement_percent(before["precision_at_10"], after["precision_at_10"])),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            _improvement_card("🎯 Recall Improvement",
                              _improvement_percent(before["recall"], after["recall"])),
            unsafe_allow_html=True
        )


def _render_refinement_comparison(selected_dataset: str) -> None:
    """Before/after comparison for models evaluated with and without query refinement."""
    # ── Refinement comparison ──────────────────────────────────────────────
    st.markdown("### 🔧 Impact of Query Refinement")

    history_resp = eval_api.get_history()

    if not (history_resp.ok and history_resp.data):
        log_failure(logger, "refinement_comparison_history", history_resp)
        st.info("No evaluation history available.")
        return

    history = history_resp.data.get("history", [])
    target_dataset = selected_dataset if selected_dataset != "All Datasets" else None

    if not target_dataset:
        st.info("📂 Please select a specific dataset from the filter above to view query refinement comparison.")
        return

    dataset_history = [h for h in history if h.get("dataset") == target_dataset]

    if not dataset_history:
        st.info(f"No evaluation history found for dataset: {target_dataset}")
        return

    with_ref, without_ref = _collect_refinement_pairs(dataset_history)
    available_models = sorted(list(set(with_ref.keys()) & set(without_ref.keys())))

    if not available_models:
        st.info("📊 No models with both refinement and non-refinement evaluations found.")
        st.caption("💡 Tip: Run evaluation with and without '✨ Enable Query Refinement' for the same model.")
        return

    selected_model = st.selectbox(
        "🤖 Select Model for Comparison",
        available_models,
        format_func=lambda x: x.upper().replace("_", " ").title(),
        key="refinement_model_selector"
    )

    if not selected_model:
        return

    before = without_ref[selected_model]
    after = with_ref[selected_model]

    st.markdown(f"#### 💡 Comparison for {selected_model.upper()} on {target_dataset} dataset")

    _render_refinement_tabs(before, after)
    _render_improvement_cards(before, after)


def _render_insights(metrics: Dict[str, dict]) -> None:
    """Best performer per metric and the MAP distribution."""
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


def _render_export(raw_metrics: Dict[str, dict]) -> None:
    """JSON export and the raw payload viewer."""
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