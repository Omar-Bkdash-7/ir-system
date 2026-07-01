"""
Index Management page — build, monitor, and rebuild TF-IDF / BM25 / Embedding indices.
"""
from __future__ import annotations

import streamlit as st

import config
from api import indexing as indexing_api
from components.index_status_badge import render_index_card
from constants import SS_BUILD_IN_PROGRESS, SS_BUILD_LOG, SS_FORCE_REBUILD


def _build_single(index_type: str, force: bool) -> None:
    label = config.RETRIEVAL_MODEL_LABELS.get(index_type, index_type)
    estimate = config.BUILD_TIME_ESTIMATES.get(index_type, "a few minutes")
    log: list = st.session_state.get(SS_BUILD_LOG, [])

    with st.status(f"Building {label} index… (est. {estimate})", expanded=True) as s:
        st.write(f"Calling indexing service — force_rebuild={force}")
        resp = indexing_api.build_index(index_type, force_rebuild=force)
        if resp.ok and isinstance(resp.data, dict):
            status_val = resp.data.get("status", "unknown")
            if status_val == "skipped":
                s.update(
                    label=f"⏭️ {label} already built — use Force Rebuild to overwrite.",
                    state="complete",
                )
                log.append(f"[SKIP] {label}: {resp.data.get('message')}")
            else:
                s.update(label=f"✅ {label} built successfully!", state="complete")
                log.append(f"[OK] {label}: {resp.data}")
            indexing_api.clear_index_status_cache()
        else:
            s.update(label=f"❌ {label} build failed", state="error")
            st.error(resp.error or "Unknown error from indexing service.")
            log.append(f"[ERR] {label}: {resp.error}")

    st.session_state[SS_BUILD_LOG] = log[-50:]  # keep last 50 entries


def render() -> None:
    st.markdown("## 🗂️ Index Management")

    # ── Build controls ─────────────────────────────────────────────────────
    hdr_c1, hdr_c2, hdr_c3 = st.columns([3, 2, 2])
    with hdr_c1:
        force = st.checkbox(
            "Force rebuild (overwrite existing)",
            value=st.session_state.get(SS_FORCE_REBUILD, False),
            key="idx_force_rebuild",
        )
        st.session_state[SS_FORCE_REBUILD] = force
    with hdr_c2:
        if st.button("🚀 Build All Indices", type="primary", use_container_width=True,
                     key="idx_build_all"):
            for itype in ("tfidf", "bm25", "embedding"):
                _build_single(itype, force)
            st.rerun()
    with hdr_c3:
        if st.button("🔄 Refresh Status", use_container_width=True, key="idx_refresh"):
            indexing_api.clear_index_status_cache()
            st.rerun()

    st.caption(
        "⏱️ Estimated build times: "
        + " · ".join(
            f"{config.RETRIEVAL_MODEL_LABELS[k]}: {v}"
            for k, v in config.BUILD_TIME_ESTIMATES.items()
        )
    )

    st.divider()

    # ── Status cards ───────────────────────────────────────────────────────
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
                _build_single(itype, force)
                st.rerun()

    st.divider()

    # ── Aggregate stats ────────────────────────────────────────────────────
    stats_resp = indexing_api.all_stats()
    if stats_resp.ok and isinstance(stats_resp.data, dict):
        with st.expander("📊 Aggregate index stats", expanded=False):
            st.json(stats_resp.data)

    # ── Build log ──────────────────────────────────────────────────────────
    log = st.session_state.get(SS_BUILD_LOG, [])
    if log:
        with st.expander(f"📋 Build log ({len(log)} entries)", expanded=False):
            st.code("\n".join(reversed(log)), language="text")
        if st.button("🗑️ Clear log", key="idx_clear_log"):
            st.session_state[SS_BUILD_LOG] = []
            st.rerun()
