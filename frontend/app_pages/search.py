"""
Search page — the main feature of the IR System.

Layout:
  1. Query bar + Refine button + Clear
  2. Preprocessing config panel (collapsible)
  3. Retrieval mode tabs: Individual | Hybrid Serial | Hybrid Parallel
  4. Search button
  5. Results (single or side-by-side columns)
  6. Query history panel
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import streamlit as st

import config
import api.gateway as gateway
import api.indexing as indexing_api
from api.client import APIResponse
from components.bm25_params import render_bm25_params
from components.preprocessing_config import (
    render_preprocessing_config,
    render_preprocessing_preset_selector,
)
from components.result_card import (
    render_refinement_banner,
    render_result_card,
    render_results_grid,
)
from constants import (
    SS_CURRENT_QUERY,
    SS_LAST_SEARCH_TIME,
    SS_PREPROCESSING_CONFIG,
    SS_QUERY_HISTORY,
    SS_REFINEMENT_ENABLED,
    SS_REFINEMENT_RESULT,
    SS_RETRIEVAL_MODE,
    SS_SEARCH_COUNT,
    SS_SEARCH_RESULTS,
    SS_SESSION_ID,
    SS_SELECTED_DATASET,
)
from ui_state import dataset_selector
from ui_style import inject_style
from utils.dataset_utils import get_available_datasets, get_dataset_display_name


# ─────────────────────────────────────────────────────────────────────────────
# Dataset metadata helper
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_to_history(query: str) -> None:
    hist: List[str] = st.session_state.get(SS_QUERY_HISTORY, [])
    if query and query not in hist:
        hist.append(query)
        if len(hist) > 10:
            hist.pop(0)
    st.session_state[SS_QUERY_HISTORY] = hist


def _active_preprocessing_config() -> dict:
    cfg = st.session_state.get(SS_PREPROCESSING_CONFIG)
    if cfg is None:
        cfg = dict(config.DEFAULT_PREPROCESSING_CONFIG)
        st.session_state[SS_PREPROCESSING_CONFIG] = cfg
    return cfg


def _run_search(query: str, mode: str, mode_cfg: dict) -> None:
    """Execute the appropriate search call and store results in session state."""
    history = st.session_state.get(SS_QUERY_HISTORY, [])
    refinement_enabled = st.session_state.get(SS_REFINEMENT_ENABLED, False)
    session_id = st.session_state.get(SS_SESSION_ID, "default")
    selected_dataset = st.session_state.get(SS_SELECTED_DATASET)

    t0 = time.perf_counter()

    with st.status("🔍 Searching…", expanded=True) as status_box:
        st.write("Preprocessing query…")

        if mode == "individual":
            enabled_models = [m for m in config.RETRIEVAL_MODELS
                              if mode_cfg.get(m, {}).get("enabled")]
            if not enabled_models:
                st.error("Select at least one model.")
                return

            results_by_model: Dict[str, List[dict]] = {}
            refinement_data: Optional[dict] = None

            for model in enabled_models:
                mcfg = mode_cfg[model]
                st.write(f"Retrieving with {config.RETRIEVAL_MODEL_LABELS[model]}…")
                resp: APIResponse = gateway.search_basic(
                    query=query,
                    model=model,
                    top_k=mcfg.get("top_k", config.DEFAULT_TOP_K),
                    bm25_k1=mcfg.get("k1", config.DEFAULT_BM25_K1),
                    bm25_b=mcfg.get("b", config.DEFAULT_BM25_B),
                    enable_refinement=refinement_enabled,
                    query_history=history[-3:],
                    dataset=selected_dataset,
                )
                if resp.ok and isinstance(resp.data, dict):
                    docs = resp.data.get("results", [])
                    disp_k = mcfg.get("display_k", config.DEFAULT_DISPLAY_K)
                    results_by_model[config.RETRIEVAL_MODEL_LABELS[model]] = docs[:disp_k]
                    if refinement_data is None:
                        refinement_data = resp.data.get("refinement")
                else:
                    st.warning(f"{config.RETRIEVAL_MODEL_LABELS[model]}: {resp.error}")

            elapsed = (time.perf_counter() - t0) * 1000
            status_box.update(label=f"✅ Done in {elapsed:.0f} ms", state="complete")
            st.session_state[SS_SEARCH_RESULTS] = {
                "mode": "individual",
                "results_by_model": results_by_model,
                "refinement": refinement_data,
                "elapsed_ms": elapsed,
                "query": query,
            }

        elif mode == "hybrid_serial":
            st.write(f"Stage 1: {mode_cfg['first_model'].upper()} retrieval…")
            resp = gateway.search_hybrid_serial(
                query=query,
                first_model=mode_cfg["first_model"],
                second_model=mode_cfg["second_model"],
                first_stage_k=mode_cfg.get("first_stage_k", config.HYBRID_SERIAL_DEFAULT_CANDIDATES),
                final_top_k=mode_cfg.get("final_top_k", config.HYBRID_SERIAL_DEFAULT_FINAL_K),
                bm25_k1=mode_cfg.get("k1", config.DEFAULT_BM25_K1),
                bm25_b=mode_cfg.get("b", config.DEFAULT_BM25_B),
                enable_refinement=refinement_enabled,
                query_history=history[-3:],
                dataset=selected_dataset,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.ok and isinstance(resp.data, dict):
                label = (f"Hybrid Serial "
                         f"[{mode_cfg['first_model'].upper()} → "
                         f"{mode_cfg['second_model'].upper()}]")
                status_box.update(label=f"✅ Done in {elapsed:.0f} ms", state="complete")
                st.session_state[SS_SEARCH_RESULTS] = {
                    "mode": "hybrid_serial",
                    "results_by_model": {label: resp.data.get("results", [])},
                    "refinement": resp.data.get("refinement"),
                    "elapsed_ms": elapsed,
                    "query": query,
                }
            else:
                status_box.update(label="❌ Search failed", state="error")
                st.error(f"Hybrid serial search failed: {resp.error}")

        elif mode == "hybrid_parallel":
            models = mode_cfg.get("models", [])
            fusion = mode_cfg.get("fusion_method", "rrf")
            st.write(f"Running {len(models)} models in parallel…")
            resp = gateway.search_hybrid_parallel(
                query=query,
                models=models,
                fusion_method=fusion,
                top_k=mode_cfg.get("final_top_k", config.HYBRID_PARALLEL_DEFAULT_FINAL_K),
                candidates_per_model=mode_cfg.get("candidates_per_model",
                                                   config.HYBRID_PARALLEL_DEFAULT_CANDIDATES),
                rrf_k=mode_cfg.get("rrf_k", config.RRF_DEFAULT_K),
                model_weights=mode_cfg.get("model_weights"),
                bm25_k1=mode_cfg.get("k1", config.DEFAULT_BM25_K1),
                bm25_b=mode_cfg.get("b", config.DEFAULT_BM25_B),
                enable_refinement=refinement_enabled,
                query_history=history[-3:],
                dataset=selected_dataset,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.ok and isinstance(resp.data, dict):
                label = (f"Hybrid Parallel "
                         f"[{' + '.join(m.upper() for m in models)}, "
                         f"{config.FUSION_METHOD_LABELS.get(fusion, fusion)}]")
                status_box.update(label=f"✅ Done in {elapsed:.0f} ms", state="complete")
                st.session_state[SS_SEARCH_RESULTS] = {
                    "mode": "hybrid_parallel",
                    "results_by_model": {label: resp.data.get("results", [])},
                    "refinement": resp.data.get("refinement"),
                    "elapsed_ms": elapsed,
                    "query": query,
                }
            else:
                status_box.update(label="❌ Search failed", state="error")
                st.error(f"Hybrid parallel search failed: {resp.error}")

    st.session_state[SS_LAST_SEARCH_TIME] = time.strftime("%H:%M:%S")
    st.session_state[SS_SEARCH_COUNT] = st.session_state.get(SS_SEARCH_COUNT, 0) + 1
    _add_to_history(query)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-sections rendered inside the mode tabs
# ─────────────────────────────────────────────────────────────────────────────

def _tab_individual() -> dict:
    """Render Individual Models tab, return mode_cfg dict."""
    st.markdown("##### Select & configure models")
    mode_cfg: dict = {}
    
    for model in config.RETRIEVAL_MODELS:
        icon = config.RETRIEVAL_MODEL_ICONS[model]
        label = config.RETRIEVAL_MODEL_LABELS[model]
        
        with st.container(border=True):
            enabled = st.checkbox(f"{icon} {label}", value=True, key=f"ind_ena_{model}")
            col_topk, col_disp = st.columns(2)
            
            with col_topk:
                top_k = st.slider(
                    "Retrieve top_k",
                    config.MIN_TOP_K, config.MAX_TOP_K,
                    config.DEFAULT_TOP_K, 10,
                    key=f"ind_topk_{model}",
                )
                
            with col_disp:
                disp_k = st.slider(
                    "Display",
                    config.MIN_DISPLAY_K, config.MAX_DISPLAY_K,
                    config.DEFAULT_DISPLAY_K, 5,
                    key=f"ind_disp_{model}",
                )
            
            mcfg: dict = {"enabled": enabled, "top_k": top_k, "display_k": disp_k}
            
            if model == "bm25":
                k1, b = render_bm25_params(key_prefix=f"ind_{model}")
                mcfg.update({"k1": k1, "b": b})
            elif model == "embedding":
                st.caption(f"Model: `{config.EMBEDDING_MODEL_NAME}` ({config.EMBEDDING_DIMENSIONS}d)")
            elif model == "tfidf":
                st.caption(f"Config: `{config.TFIDF_DESCRIPTION}`")
            
            mode_cfg[model] = mcfg
            
    return mode_cfg

def _tab_hybrid_serial() -> dict:
    """Render Hybrid Serial tab, return mode_cfg dict."""
    st.markdown("##### Two-stage re-ranking pipeline")
    model_options = config.RETRIEVAL_MODELS

    c1, c2 = st.columns(2)
    with c1:
        first = st.selectbox(
            "1st Stage (fast retrieval)",
            model_options,
            index=model_options.index("bm25"),
            key="hs_first",
            format_func=lambda x: f"{config.RETRIEVAL_MODEL_ICONS[x]} {config.RETRIEVAL_MODEL_LABELS[x]}",
        )
        first_k = st.slider(
            "Candidates",
            config.MIN_SERIAL_CANDIDATES, config.MAX_SERIAL_CANDIDATES,
            config.HYBRID_SERIAL_DEFAULT_CANDIDATES, 100,
            key="hs_first_k",
        )
    with c2:
        second_opts = [m for m in model_options if m != first]
        second = st.selectbox(
            "2nd Stage (re-ranker)",
            second_opts,
            index=second_opts.index("embedding") if "embedding" in second_opts else 0,
            key="hs_second",
            format_func=lambda x: f"{config.RETRIEVAL_MODEL_ICONS[x]} {config.RETRIEVAL_MODEL_LABELS[x]}",
        )
        final_k = st.slider(
            "Final results",
            config.MIN_SERIAL_FINAL_K, config.MAX_SERIAL_FINAL_K,
            config.HYBRID_SERIAL_DEFAULT_FINAL_K, 10,
            key="hs_final_k",
        )

    # Pipeline diagram
    st.markdown(
        f'<div class="pipeline-row">'
        f'<div class="pipeline-box">{config.RETRIEVAL_MODEL_ICONS[first]} '
        f'{config.RETRIEVAL_MODEL_LABELS[first]}<br>'
        f'<small>{first_k} candidates</small></div>'
        f'<span class="pipeline-arrow">→</span>'
        f'<div class="pipeline-box pipeline-box--accent">'
        f'{config.RETRIEVAL_MODEL_ICONS[second]} '
        f'{config.RETRIEVAL_MODEL_LABELS[second]}<br>'
        f'<small>re-rank</small></div>'
        f'<span class="pipeline-arrow">→</span>'
        f'<div class="pipeline-box pipeline-box--final">Top {final_k}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cfg: dict = {
        "first_model": first,
        "second_model": second,
        "first_stage_k": first_k,
        "final_top_k": final_k,
        "k1": config.DEFAULT_BM25_K1,
        "b": config.DEFAULT_BM25_B,
    }
    if first == "bm25" or second == "bm25":
        with st.expander("BM25 parameters", expanded=False):
            k1, b = render_bm25_params(key_prefix="hs_bm25")
            cfg.update({"k1": k1, "b": b})

    return cfg


def _tab_hybrid_parallel() -> dict:
    """Render Hybrid Parallel tab, return mode_cfg dict."""
    inject_style("dataframe_tags")

    st.markdown("##### Parallel fusion across multiple models")

    models = st.multiselect(
        "Models to run in parallel (min 2)",
        config.RETRIEVAL_MODELS,
        default=["tfidf", "bm25"],
        key="hp_models",
        format_func=lambda x: f"{config.RETRIEVAL_MODEL_ICONS[x]} {config.RETRIEVAL_MODEL_LABELS[x]}",
    )
    if len(models) < 2:
        st.error("⚠️ Select at least 2 models.")

    fusion_options = config.FUSION_METHODS
    fusion = st.selectbox(
        "Fusion method",
        fusion_options,
        key="hp_fusion",
        format_func=lambda x: config.FUSION_METHOD_LABELS.get(x, x),
    )
    st.caption(f"Formula: `{config.FUSION_METHOD_FORMULAS.get(fusion, '')}`")

    c1, c2 = st.columns(2)
    with c1:
        candidates = st.slider(
            "Candidates per model",
            config.MIN_PARALLEL_CANDIDATES, config.MAX_PARALLEL_CANDIDATES,
            config.HYBRID_PARALLEL_DEFAULT_CANDIDATES, 100,
            key="hp_candidates",
        )
    with c2:
        final_k = st.slider(
            "Final results after fusion",
            config.MIN_PARALLEL_FINAL_K, config.MAX_PARALLEL_FINAL_K,
            config.HYBRID_PARALLEL_DEFAULT_FINAL_K, 10,
            key="hp_final_k",
        )

    cfg: dict = {
        "models": models,
        "fusion_method": fusion,
        "candidates_per_model": candidates,
        "final_top_k": final_k,
        "k1": config.DEFAULT_BM25_K1,
        "b": config.DEFAULT_BM25_B,
    }

    if fusion == "rrf":
        cfg["rrf_k"] = st.slider(
            "RRF k",
            config.MIN_RRF_K, config.MAX_RRF_K,
            config.RRF_DEFAULT_K, 5,
            key="hp_rrf_k",
            help="Higher k → less rank-difference sensitivity",
        )

    if fusion == "weighted" and len(models) >= 2:
        st.markdown("**Weight per model** (must sum to 1.0)")
        
        # Auto-balance callback
        def auto_balance_weights():
            balanced = round(1.0 / len(models), 4)
            for m in models:
                st.session_state[f"hp_w_{m}"] = balanced
        
        weights: dict = {}
        wcols = st.columns(len(models))
        default_w = round(1.0 / len(models), 2)
        for col, m in zip(wcols, models):
            with col:
                weights[m] = st.slider(
                    config.RETRIEVAL_MODEL_LABELS[m],
                    0.0, 1.0, default_w, 0.01,
                    key=f"hp_w_{m}",
                )
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total:.2f} — must equal 1.0")
            st.button("⚖️ Auto-balance", key="hp_autobalance", on_click=auto_balance_weights)
        else:
            st.success(f"✅ Sum = {total:.2f}")
        cfg["model_weights"] = weights

    if "bm25" in models:
        with st.expander("BM25 parameters", expanded=False):
            k1, b = render_bm25_params(key_prefix="hp_bm25")
            cfg.update({"k1": k1, "b": b})

    # Fusion diagram
    if len(models) >= 2:
        sources_html = "".join(
            f'<div class="pipeline-box">{config.RETRIEVAL_MODEL_ICONS[m]} '
            f'{config.RETRIEVAL_MODEL_LABELS[m]}</div>'
            for m in models
        )
        st.markdown(
            f'<div class="fusion-wrap">'
            f'<div class="fusion-sources">{sources_html}</div>'
            f'<div class="fusion-arrow-down">↓</div>'
            f'<div class="fusion-box">🔀 {config.FUSION_METHOD_LABELS.get(fusion, fusion)}</div>'
            f'<div class="fusion-arrow-down">↓</div>'
            f'<div class="pipeline-box pipeline-box--final">Top {final_k}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main page render function
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## 🔍 Search")

    inject_style("search_layout")

    # ── Dataset selector ───────────────────────────────────────────────────
    available_datasets = get_available_datasets()
    
    st.caption("Choose a dataset with all indices built")
    
    def on_dataset_change():
        """A new dataset invalidates the results of the previous one."""
        if st.session_state.dataset_selector:
            st.session_state[SS_SEARCH_RESULTS] = None
    
    if available_datasets:
        dataset_selector(
            available_datasets,
            key="dataset_selector",
            label="Dataset",
            placeholder="Select a dataset...",
            format_func=get_dataset_display_name,
            help="Only datasets with all indices (TF-IDF, BM25, Embedding) built are shown",
            label_visibility="collapsed",
            on_change=on_dataset_change,
        )
    else:
        st.selectbox(
            "Dataset",
            options=[],
            placeholder="No datasets with all indices available",
            key="dataset_selector_empty",
            help="No datasets available. Download a dataset and build all indices (TF-IDF, BM25, Embedding) to enable search.",
            disabled=True,
            label_visibility="collapsed",
        )
    
    # ── Dataset Information Cards ──────────────────────────────────────────
    if st.session_state.get(SS_SELECTED_DATASET):
        selected_metadata = indexing_api.get_dataset_metadata(st.session_state[SS_SELECTED_DATASET])
        
        if selected_metadata:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Dataset", st.session_state[SS_SELECTED_DATASET])
                
                with col2:
                    doc_count = selected_metadata.get('doc_count', 0)
                    st.metric("Docs", f"{doc_count:,}")
                
                with col3:
                    query_count = selected_metadata.get('query_count', 0)
                    st.metric("Queries", f"{query_count:,}")
                
                with col4:
                    qrel_count = selected_metadata.get('qrel_count', selected_metadata.get('qrels_count', 0))
                    st.metric("Qrels", f"{qrel_count:,}")
    
    st.divider()

    # ── Query input bar ────────────────────────────────────────────────────
    qcol, rcol = st.columns([5, 1])
    with qcol:
        query = st.text_input(
            "Query",
            value=st.session_state.get(SS_CURRENT_QUERY, ""),
            placeholder="Enter your search query…",
            label_visibility="collapsed",
            key="search_query_input",
        )
        # Store query value for next render
        if query != st.session_state.get(SS_CURRENT_QUERY, ""):
            st.session_state[SS_CURRENT_QUERY] = query

    with rcol:
        if st.button("✨ Refine", use_container_width=True, key="btn_refine"):
            if query.strip():
                with st.spinner("Refining…"):
                    resp = gateway.refine_query(
                        query,
                        st.session_state.get(SS_QUERY_HISTORY, [])[-3:],
                        enabled=True,
                        session_id=st.session_state.get(SS_SESSION_ID, "default"),
                        dataset=st.session_state.get(SS_SELECTED_DATASET),
                    )
                    if resp.ok:
                        st.session_state[SS_REFINEMENT_RESULT] = resp.data
                    else:
                        st.warning(f"Refinement unavailable: {resp.error}")
            else:
                st.warning("Enter a query first.")

    # ── Refinement preview ─────────────────────────────────────────────────
    # Check if we should show refinement expander
    show_refinement = st.session_state.get(SS_REFINEMENT_RESULT) is not None
    
    if show_refinement:
        refinement_result = st.session_state[SS_REFINEMENT_RESULT]
        if refinement_result.get("applied"):
            with st.expander("✨ Refinement Suggestion — click to review", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original**")
                    st.info(refinement_result.get("original_query", ""))
                with c2:
                    st.markdown("**Suggested**")
                    st.success(refinement_result.get("refined_query", ""))
                
                a1, a2 = st.columns(2)
                with a1:
                    if st.button("✅ Accept refined query", use_container_width=True, key="btn_accept_ref"):
                        # Update query and clear refinement
                        st.session_state[SS_CURRENT_QUERY] = refinement_result.get("refined_query", query)
                        st.session_state[SS_REFINEMENT_RESULT] = None
                        st.rerun()
                with a2:
                    if st.button("❌ Keep original", use_container_width=True, key="btn_reject_ref"):
                        # Just clear refinement
                        st.session_state[SS_REFINEMENT_RESULT] = None
                        st.rerun()

    # ── Preprocessing config ────────────────────────────────────────────────
    with st.expander("⚙️ Preprocessing Configuration", expanded=False):
        render_preprocessing_preset_selector()
        st.divider()
        prep_cfg = render_preprocessing_config(key_prefix="search_prep")

        prev_col, _ = st.columns([1, 3])
        with prev_col:
            if st.button("👁️ Preview tokens", key="btn_preview_prep") and query.strip():
                from api.preprocessing import preprocess
                resp = preprocess(query, prep_cfg)
                if resp.ok and isinstance(resp.data, dict):
                    tokens = resp.data.get("tokens", [])
                    chips = "".join(f'<span class="chip">{t}</span>' for t in tokens)
                    st.markdown(
                        f'<div class="chip-row">{chips}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{len(tokens)} tokens")
                else:
                    st.warning(f"Preview failed: {resp.error}")

    # ── Query refinement toggle ─────────────────────────────────────────────
    st.session_state[SS_REFINEMENT_ENABLED] = st.toggle(
        "✨ Enable Query Refinement on search",
        value=st.session_state.get(SS_REFINEMENT_ENABLED, False),
        key="search_refinement_toggle",
        help="Automatically applies spelling correction, synonyms, and history-aware boosting.",
    )

    # ── Retrieval mode segmented control ───────────────────────────────────
    st.markdown("### Retrieval Mode")
    
    # Callback to update mode immediately
    def _on_mode_change():
        st.session_state[SS_RETRIEVAL_MODE] = st.session_state["mode_radio"]
    
    # Get current mode
    current_mode = st.session_state.get(SS_RETRIEVAL_MODE, "individual")
    
    # Segmented control using radio buttons with custom styling
    mode_choice = st.radio(
        "Active mode",
        ["individual", "hybrid_serial", "hybrid_parallel"],
        index=["individual", "hybrid_serial", "hybrid_parallel"].index(current_mode),
        horizontal=True,
        key="mode_radio",
        label_visibility="collapsed",
        format_func=lambda x: {
            "individual": "📊 Individual Models",
            "hybrid_serial": "🔗 Hybrid Serial",
            "hybrid_parallel": "🔀 Hybrid Parallel"
        }[x],
        on_change=_on_mode_change
    )

    # Render the appropriate configuration panel based on selected mode
    if mode_choice == "individual":
        active_mode_cfg = _tab_individual()
        active_mode = "individual"
    elif mode_choice == "hybrid_serial":
        active_mode_cfg = _tab_hybrid_serial()
        active_mode = "hybrid_serial"
    else:
        active_mode_cfg = _tab_hybrid_parallel()
        active_mode = "hybrid_parallel"

    # ── Search button ───────────────────────────────────────────────────────
    st.divider()
    if st.button("🚀 Search", type="primary", use_container_width=True, key="btn_search"):
        if not st.session_state.get(SS_SELECTED_DATASET):
            st.error("⚠️ Please select a dataset first before searching.")
        elif not query.strip():
            st.error("❌ Enter a query.")
        else:
            _run_search(query.strip(), active_mode, active_mode_cfg)
            st.rerun()

    # ── Results ─────────────────────────────────────────────────────────────
    results_state = st.session_state.get(SS_SEARCH_RESULTS)
    if results_state:
        elapsed = results_state.get("elapsed_ms", 0)
        results_by_model = results_state.get("results_by_model", {})
        total_docs = sum(len(v) for v in results_by_model.values())

        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("⏱️ Latency", f"{elapsed:.0f} ms")
        with m2:
            st.metric("📄 Results", total_docs)
        with m3:
            st.metric("🎯 Mode", results_state.get("mode", "—").replace("_", " ").title())

        render_refinement_banner(results_state.get("refinement") or {})

        if len(results_by_model) > 1:
            render_results_grid(results_by_model)
        else:
            for model_label, docs in results_by_model.items():
                st.markdown(f"#### {model_label} — {len(docs)} results")
                for idx, doc in enumerate(docs, 1):
                    render_result_card(doc, idx, model_label=model_label)

    # ── Query history ───────────────────────────────────────────────────────
    history = st.session_state.get(SS_QUERY_HISTORY, [])
    if history:
        with st.expander(f"🕑 Query History ({len(history)})", expanded=False):
            for i, q in enumerate(reversed(history)):
                hc1, hc2 = st.columns([4, 1])
                with hc1:
                    st.text(q)
                with hc2:
                    if st.button("↩ Reuse", key=f"hist_reuse_{i}"):
                        st.session_state[SS_CURRENT_QUERY] = q
                        st.rerun()
