"""
Dataset & Preprocessing overview page.
Covers: dataset info card, ir_datasets download/validate button, metadata viewer.
"""
from __future__ import annotations

import streamlit as st

import config
from api import gateway
from api.indexing import validate_dataset
from constants import SS_DATASET_INFO, SS_DATASET_VALIDATION_RESULT, SS_VALIDATION_IN_PROGRESS


def render() -> None:
    st.markdown("## 📊 Dataset")

    # ── Dataset info card ─────────────────────────────────────────────────
    info = gateway.get_dataset_info_cached()

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        if info:
            with c1:
                st.metric("Dataset", info.get("dataset_name", config.DATASET_NAME))
            with c2:
                st.metric("Documents", f"{info.get('doc_count', config.DATASET_DOC_COUNT):,}")
            with c3:
                st.metric("Test Queries", f"{info.get('query_count', config.DATASET_QUERY_COUNT_APPROX):,}")
            with c4:
                st.metric("Qrels", f"{info.get('qrel_count', config.DATASET_QREL_COUNT_APPROX):,}")
        else:
            with c1:
                st.metric("Dataset", config.DATASET_NAME)
            with c2:
                st.metric("Documents", f"{config.DATASET_DOC_COUNT:,}")
            with c3:
                st.metric("Test Queries", f"~{config.DATASET_QUERY_COUNT_APPROX:,}")
            with c4:
                st.metric("Qrels", f"~{config.DATASET_QREL_COUNT_APPROX:,}")
            st.warning(
                "⚠️ Metadata file not found on backend — "
                "run validation below to generate it."
            )

    st.divider()

    # ── Validate / download dataset ───────────────────────────────────────
    st.markdown("### Validate Dataset (ir_datasets)")
    st.markdown(
        "Triggers the indexing service to load `beir/quora` via `ir_datasets`, "
        "validate document/query/qrel counts, and write `data/metadata.json`. "
        "**This can take several minutes** on first run while the dataset is downloaded."
    )

    v_col1, v_col2 = st.columns([2, 3])
    with v_col1:
        dataset_id = st.text_input(
            "Dataset ID",
            value=config.DATASET_NAME,
            key="dataset_id_input",
        )
    with v_col2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("⬇️ Download & Validate", type="primary", use_container_width=True,
                     key="btn_validate_dataset"):
            st.session_state[SS_VALIDATION_IN_PROGRESS] = True
            with st.status("Validating dataset… this may take a few minutes.", expanded=True) as s:
                st.write(f"Calling `POST /validate?dataset_id={dataset_id}`")
                resp = validate_dataset(dataset_id)
                if resp.ok and isinstance(resp.data, dict):
                    s.update(label="✅ Dataset validated!", state="complete")
                    st.session_state[SS_DATASET_VALIDATION_RESULT] = resp.data
                    gateway.clear_dataset_info_cache()
                else:
                    s.update(label="❌ Validation failed", state="error")
                    st.error(resp.error or "Unknown error from indexing service.")
                    st.session_state[SS_DATASET_VALIDATION_RESULT] = {"error": resp.error}
            st.session_state[SS_VALIDATION_IN_PROGRESS] = False

    val_result = st.session_state.get(SS_DATASET_VALIDATION_RESULT)
    if val_result:
        if "error" not in val_result:
            st.success("Dataset validated and metadata saved.")
            with st.expander("📄 Validation response", expanded=False):
                st.json(val_result)
        else:
            st.error(f"Validation error: {val_result.get('error')}")

    st.divider()

    # ── Full metadata viewer ──────────────────────────────────────────────
    st.markdown("### Metadata")
    if st.button("🔄 Refresh metadata", key="btn_refresh_meta"):
        gateway.clear_dataset_info_cache()
        st.rerun()

    fresh_resp = gateway.get_dataset_info()
    if fresh_resp.ok and isinstance(fresh_resp.data, dict):
        with st.expander("📄 data/metadata.json", expanded=True):
            st.json(fresh_resp.data)
    else:
        st.info("Metadata not available — validate the dataset first.")
