"""
Index Management page — build, monitor, and rebuild TF-IDF / BM25 / Embedding indices.
"""
from __future__ import annotations

import streamlit as st

import config
import api.indexing as indexing_api
import api.gateway as gateway
from components.index_status_badge import render_index_card
from constants import SS_BUILD_IN_PROGRESS, SS_BUILD_LOG, SS_FORCE_REBUILD


def _get_downloaded_datasets() -> list[str]:
    """
    Get list of downloaded datasets from indexing service.
    Returns list of dataset IDs like ['beir/quora', 'cranfield', etc.]
    """
    resp = indexing_api.list_local_datasets()
    
    if not resp.ok:
        return []
    
    datasets_info = resp.data.get("datasets", []) if isinstance(resp.data, dict) else []
    
    # Extract just the dataset IDs from the full info
    dataset_ids = [ds.get("dataset_id") for ds in datasets_info if ds.get("dataset_id")]
    
    return sorted(dataset_ids)


def _render_preprocessing_config() -> dict | None:
    """Render preprocessing configuration options and return config dict."""
    
    with st.expander("⚙️ Preprocessing Configuration", expanded=False):
        st.caption("Configure how documents are preprocessed before indexing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Text Normalization**")
            normalize_unicode = st.checkbox(
                "Normalize Unicode",
                value=config.DEFAULT_PREPROCESSING_CONFIG["normalize_unicode"],
                help="Convert text to normalized Unicode form"
            )
            lowercase = st.checkbox(
                "Lowercase",
                value=config.DEFAULT_PREPROCESSING_CONFIG["lowercase"],
                help="Convert all text to lowercase"
            )
            remove_punctuation = st.checkbox(
                "Remove Punctuation",
                value=config.DEFAULT_PREPROCESSING_CONFIG["remove_punctuation"],
                help="Remove punctuation marks"
            )
            keep_numbers = st.checkbox(
                "Keep Numbers",
                value=config.DEFAULT_PREPROCESSING_CONFIG["keep_numbers"],
                help="Keep numeric tokens"
            )
        
        with col2:
            st.markdown("**Token Processing**")
            remove_stopwords = st.checkbox(
                "Remove Stopwords",
                value=config.DEFAULT_PREPROCESSING_CONFIG["remove_stopwords"],
                help="Remove common English stopwords"
            )
            use_lemmatization = st.checkbox(
                "Use Lemmatization",
                value=config.DEFAULT_PREPROCESSING_CONFIG["use_lemmatization"],
                help="Reduce words to base form (better, slower)"
            )
            use_stemming = st.checkbox(
                "Use Stemming",
                value=config.DEFAULT_PREPROCESSING_CONFIG["use_stemming"],
                help="Reduce words to stem (faster, less accurate)"
            )
            use_spacy_tokenizer = st.checkbox(
                "Use spaCy Tokenizer",
                value=config.DEFAULT_PREPROCESSING_CONFIG["use_spacy_tokenizer"],
                help="Use spaCy for advanced tokenization"
            )
        
        st.markdown("**Token Length Filters**")
        col3, col4 = st.columns(2)
        with col3:
            min_token_length = st.number_input(
                "Min Token Length",
                min_value=1,
                max_value=10,
                value=config.DEFAULT_PREPROCESSING_CONFIG["min_token_length"],
                help="Minimum character length for tokens"
            )
        with col4:
            max_token_length = st.number_input(
                "Max Token Length (0 = no limit)",
                min_value=0,
                max_value=100,
                value=0,
                help="Maximum character length for tokens"
            )
        
        apply_custom = st.checkbox(
            "Apply custom preprocessing",
            value=False,
            help="Use the configuration above instead of defaults"
        )
        
        if apply_custom:
            return {
                "normalize_unicode": normalize_unicode,
                "lowercase": lowercase,
                "remove_punctuation": remove_punctuation,
                "remove_stopwords": remove_stopwords,
                "min_token_length": min_token_length,
                "max_token_length": max_token_length if max_token_length > 0 else None,
                "use_lemmatization": use_lemmatization,
                "use_stemming": use_stemming,
                "keep_numbers": keep_numbers,
                "use_spacy_tokenizer": use_spacy_tokenizer,
                "custom_stopwords": []
            }
    
    return None


def _build_single(index_type: str, force: bool, preprocessing_config: dict | None = None) -> None:
    label = config.RETRIEVAL_MODEL_LABELS.get(index_type, index_type)
    estimate = config.BUILD_TIME_ESTIMATES.get(index_type, "a few minutes")
    log: list = st.session_state.get(SS_BUILD_LOG, [])

    with st.status(f"Building {label} index… (est. {estimate})", expanded=True) as s:
        prep_msg = " with custom preprocessing" if preprocessing_config else ""
        st.write(f"📋 Calling indexing service — force_rebuild={force}{prep_msg}")
        
        progress_container = st.empty()
        status_container = st.empty()
        
        stages = {
            "tfidf": [
                (10, "Loading preprocessed documents..."),
                (30, "Building vocabulary..."),
                (50, "Computing TF-IDF weights..."),
                (70, "Creating sparse matrix..."),
                (90, "Saving index files..."),
                (100, "Complete!")
            ],
            "bm25": [
                (10, "Loading preprocessed documents..."),
                (25, "Building inverted index..."),
                (45, "Computing document frequencies..."),
                (65, "Calculating IDF scores..."),
                (85, "Finalizing index structure..."),
                (100, "Complete!")
            ],
            "embedding": [
                (5, "Loading sentence transformer model..."),
                (10, "Model loaded, preparing documents..."),
                (30, "Generating embeddings (batch 1/10)..."),
                (50, "Generating embeddings (batch 5/10)..."),
                (70, "Generating embeddings (batch 8/10)..."),
                (90, "Saving embeddings..."),
                (100, "Complete!")
            ]
        }
        
        current_stages = stages.get(index_type, [(50, "Building..."), (100, "Complete!")])
        
        import threading
        import time
        
        result_holder = {"done": False, "resp": None}
        
        def _build_thread():
            try:
                result_holder["resp"] = indexing_api.build_index(
                    index_type, 
                    force_rebuild=force, 
                    preprocessing_config=preprocessing_config
                )
            except Exception as e:
                result_holder["resp"] = type('obj', (object,), {
                    'ok': False, 
                    'error': str(e),
                    'data': None
                })()
            finally:
                result_holder["done"] = True
        
        build_thread = threading.Thread(target=_build_thread, daemon=True)
        build_thread.start()
        
        stage_idx = 0
        start_time = time.time()
        
        while not result_holder["done"] and stage_idx < len(current_stages):
            pct, msg = current_stages[stage_idx]
            progress_container.progress(pct, text=f"{pct}% - {msg}")
            status_container.caption(f"⏳ {msg}")
            
            elapsed = int(time.time() - start_time)
            if stage_idx > 0:
                estimated_total = (elapsed / (stage_idx / len(current_stages)))
                remaining = int(estimated_total - elapsed)
                status_container.caption(
                    f"⏳ {msg} | Elapsed: {elapsed}s | Est. remaining: {remaining}s"
                )
            
            time.sleep(2)
            stage_idx += 1
        
        build_thread.join(timeout=5)
        
        if not result_holder["done"]:
            progress_container.progress(100, text="Waiting for completion...")
            build_thread.join()
        
        progress_container.progress(100, text="100% - Finalizing...")
        status_container.empty()
        
        resp = result_holder["resp"]
        
        if resp and resp.ok and isinstance(resp.data, dict):
            status_val = resp.data.get("status", "unknown")
            index_size = resp.data.get("index_size_mb", "N/A")
            vocab_size = resp.data.get("vocab_size", "N/A")
            
            if status_val == "skipped":
                s.update(
                    label=f"⏭️ {label} already built — use Force Rebuild to overwrite.",
                    state="complete",
                )
                log.append(f"[SKIP] {label}: {resp.data.get('message')}")
            else:
                success_msg = f"✅ {label} built successfully!"
                if index_size != "N/A":
                    success_msg += f" | Size: {index_size} MB"
                if vocab_size != "N/A":
                    success_msg += f" | Terms: {vocab_size:,}"
                
                s.update(label=success_msg, state="complete")
                st.balloons()
                log.append(f"[OK] {label}: size={index_size}MB, vocab={vocab_size}")
            
            indexing_api.clear_index_status_cache()
        else:
            error_msg = resp.error if resp else "Unknown error from indexing service."
            s.update(label=f"❌ {label} build failed", state="error")
            st.error(error_msg)
            log.append(f"[ERR] {label}: {error_msg}")

    st.session_state[SS_BUILD_LOG] = log[-50:]


def _on_dataset_change():
    """Callback when dataset selection changes."""
    selected = st.session_state.get("dataset_selector")
    previous = st.session_state.get("_previous_dataset")
    
    if selected and selected != previous:
        resp = indexing_api.validate_dataset(selected)
        if resp.ok:
            gateway.clear_dataset_info_cache()
            indexing_api.clear_index_status_cache()
            st.session_state["_previous_dataset"] = selected
            st.session_state["dataset_change_success"] = selected
            st.session_state["dataset_needs_rerun"] = True
        else:
            st.session_state["dataset_change_error"] = resp.error


def render() -> None:
    st.markdown("## 🗂️ Index Management")
    
    # Check if we need to rerun after dataset change
    if st.session_state.get("dataset_needs_rerun"):
        del st.session_state["dataset_needs_rerun"]
        st.rerun()
    
    # Show feedback messages from dataset change
    if "dataset_change_success" in st.session_state:
        st.success(f"✅ Dataset switched to: {st.session_state['dataset_change_success']}")
        del st.session_state["dataset_change_success"]
    
    if "dataset_change_error" in st.session_state:
        st.error(f"❌ Validation failed: {st.session_state['dataset_change_error']}")
        del st.session_state["dataset_change_error"]
    
    # Dataset selection
    with st.container(border=True):
        st.markdown("### 📊 Active Dataset")
        
        # Get current validated dataset
        info = gateway.get_dataset_info_cached()
        current_dataset = info.get("dataset_name", config.DATASET_NAME) if info else config.DATASET_NAME
        
        # Get available datasets
        available_datasets = _get_downloaded_datasets()
        
        # Dataset selector
        if available_datasets:
            # Find current index for selectbox
            try:
                default_idx = available_datasets.index(current_dataset) if current_dataset in available_datasets else 0
            except (ValueError, TypeError):
                default_idx = 0
            
            # Initialize previous dataset tracker
            if "_previous_dataset" not in st.session_state:
                st.session_state["_previous_dataset"] = current_dataset
            
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=available_datasets,
                index=default_idx,
                key="dataset_selector",
                on_change=_on_dataset_change,
                help="Choose a dataset - will automatically validate and switch"
            )
            
            st.caption(f"💡 **Available datasets:** {len(available_datasets)} | Currently active: **{current_dataset}**")
        else:
            st.warning("⚠️ No datasets found in `data/datasets/.ir_datasets/`")
            st.info("📥 Go to **Dataset** page to download datasets first")
        
        # Show current active dataset info
        if info and info.get("validation_status") == "success":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Dataset", current_dataset)
            
            with col2:
                st.metric("Documents", f"{info.get('doc_count', 0):,}")
            
            with col3:
                st.metric("Queries", f"{info.get('query_count', 0):,}")
            
            with col4:
                st.metric("Qrels", f"{info.get('qrel_count', info.get('qrels_count', 0)):,}")
            
            # Calculate storage info based on actual dataset size
            doc_count = info.get('doc_count', 0)
            
            # Estimate sizes based on document count (linear approximation)
            # Base estimates from 500K docs
            scale_factor = doc_count / 500_000
            est_tfidf_mb = int(200 * scale_factor)
            est_bm25_mb = int(150 * scale_factor)
            est_embedding_mb = int(800 * scale_factor)
            
            # Estimate build times (sublinear growth)
            import math
            time_factor = math.sqrt(scale_factor)  # Square root for sublinear growth
            est_tfidf_min = max(1, int(1 * time_factor))
            est_bm25_min = max(1, int(2 * time_factor))
            est_embedding_min = max(2, int(15 * time_factor))
            
            # Storage path info
            dataset_slug = current_dataset.replace('/', '_').replace('-', '_')
            st.caption(
                f"💾 **Index storage:** `data/indices/{dataset_slug}/` "
                f"(TF-IDF: ~{est_tfidf_mb}MB · BM25: ~{est_bm25_mb}MB · Embeddings: ~{est_embedding_mb}MB)"
            )
            st.caption(
                f"⏱️ **Estimated build times:** "
                f"TF-IDF: ~{est_tfidf_min} min · BM25: ~{est_bm25_min} min · Embedding: ~{est_embedding_min} min"
            )

    st.divider()

    preprocessing_config = _render_preprocessing_config()
    
    st.divider()

    hdr_c1, hdr_c2, hdr_c3 = st.columns([3, 2, 2])
    with hdr_c1:
        force = st.checkbox(
            "Force rebuild (overwrite existing)",
            value=st.session_state.get(SS_FORCE_REBUILD, False),
            key="idx_force_rebuild",
        )
        st.session_state[SS_FORCE_REBUILD] = force
    with hdr_c2:
        # Enable button if we have available datasets (they're all valid)
        has_available_datasets = len(available_datasets) > 0 if available_datasets else False
        if st.button("🚀 Build All Indices", type="primary", use_container_width=True,
                     key="idx_build_all", disabled=not has_available_datasets):
            for itype in ("tfidf", "bm25", "embedding"):
                _build_single(itype, force, preprocessing_config)
            st.rerun()
    with hdr_c3:
        if st.button("🔄 Refresh Status", use_container_width=True, key="idx_refresh"):
            indexing_api.clear_index_status_cache()
            st.rerun()

    st.divider()

    st.markdown("### 📦 Index Status")
    st.caption("Build indices for the active dataset. Indices are stored as optimized files for fast retrieval.")

    # Always show index cards, just disable actions if no validated dataset
    cols = st.columns(3)
    for col, itype in zip(cols, ("tfidf", "bm25", "embedding")):
        with col:
            resp = indexing_api.index_status(itype)
            status_data = resp.data if resp.ok and isinstance(resp.data, dict) else {}
            if not resp.ok:
                status_data = {"status": "unknown", "error": resp.error}
            rebuild_clicked = render_index_card(
                index_type=itype,
                status_data=status_data,
                on_rebuild_key=f"idx_rebuild_{itype}",
            )
            if rebuild_clicked:
                _build_single(itype, force, preprocessing_config)
                st.rerun()

    st.divider()

    stats_resp = indexing_api.all_stats()
    if stats_resp.ok and isinstance(stats_resp.data, dict):
        with st.expander("📊 Aggregate index stats", expanded=False):
            st.json(stats_resp.data)

    log = st.session_state.get(SS_BUILD_LOG, [])
    if log:
        with st.expander(f"📋 Build log ({len(log)} entries)", expanded=False):
            st.code("\n".join(reversed(log)), language="text")
        if st.button("🗑️ Clear log", key="idx_clear_log"):
            st.session_state[SS_BUILD_LOG] = []
            st.rerun()
