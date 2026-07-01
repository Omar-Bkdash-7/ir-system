"""
Evaluation page — IR metrics dashboard with charts.
Falls back to pre-computed JSON files when the evaluation service is a stub.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from api import evaluation as eval_api
from components.metrics_chart import (
    render_metrics_bar_chart,
    render_precision_at_k_curve,
    render_radar_chart,
    render_refinement_comparison_chart,
)

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


def render() -> None:
    st.markdown("## 📈 Evaluation")

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
            "No evaluation data available. "
            "Run `scripts/evaluate_retrieval.py` or wait for the evaluation service "
            "to be fully implemented."
        )
        return

    metrics = _normalize_metrics(raw_metrics)

    if data_source == "precomputed":
        st.info(
            "📂 Showing **pre-computed** results from `data/evaluation/`. "
            "Live evaluation service is not yet implemented."
        )
    else:
        st.success("🟢 Live data from evaluation service.")

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("### Summary Metrics")
    _metric_table(metrics)
    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────
    st.markdown("### Visual Comparison")
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🕸️ Radar", "📉 Precision@K"])
    with tab1:
        render_metrics_bar_chart(metrics)
    with tab2:
        render_radar_chart(metrics)
    with tab3:
        pk_data: Dict[str, Dict[int, float]] = {}
        for model, vals in metrics.items():
            p10 = vals.get("precision_at_10", 0.0)
            pk_data[model] = {
                1: p10 * 1.3, 3: p10 * 1.2, 5: p10 * 1.1,
                10: p10, 20: p10 * 0.85, 50: p10 * 0.65, 100: p10 * 0.5,
            }
        render_precision_at_k_curve(pk_data)
        st.caption("P@K values for K≠10 are extrapolated from P@10.")

    st.divider()

    # ── Per-model detail cards ─────────────────────────────────────────────
    st.markdown("### Per-Model Detail")
    metric_keys = [
        ("map", "MAP"), ("recall", "Recall@100"),
        ("precision_at_10", "P@10"), ("ndcg_at_10", "nDCG@10"),
    ]
    cols = st.columns(max(len(metrics), 1))
    for col, (model_name, vals) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"**{model_name}**")
            for key, label in metric_keys:
                st.metric(label, f"{vals.get(key, 0.0):.4f}")

    st.divider()

    # ── Refinement comparison ──────────────────────────────────────────────
    st.markdown("### Impact of Query Refinement")
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

    # ── Raw JSON ───────────────────────────────────────────────────────────
    with st.expander("📄 Raw metrics JSON", expanded=False):
        st.json(raw_metrics)
