"""
Sidebar component — always visible.
"""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

import config
import api.gateway as gateway
import api.health as health_api
import api.indexing as indexing_api
from constants import (
    SS_AUTO_REFRESH,
    SS_AUTO_REFRESH_INTERVAL,
    SS_LAST_REFRESH_TIME,
    SS_SELECTED_PAGE,
)

_PAGES = [
    {"key": "search", "label": "Search", "icon": "🔍"},
    {"key": "dataset", "label": "Dataset", "icon": "📊"},
    {"key": "indexing", "label": "Indexing", "icon": "🗂️"},
    {"key": "inverted_index", "label": "Inverted Index", "icon": "🗃️"},
    {"key": "preprocessing", "label": "Preprocessing", "icon": "⚙️"},
    {"key": "evaluation", "label": "Evaluation", "icon": "📈"},
    {"key": "query_refinement", "label": "Query Refinement", "icon": "🔬"},
    {"key": "system_status", "label": "System Status", "icon": "🖥️"},
]

_STATUS_EMOJI = {True: "🟢", False: "🔴", None: "🟡"}


def render_sidebar() -> None:
    """Renders the full sidebar. Navigation lives in session_state[SS_SELECTED_PAGE]."""
    with st.sidebar:

        # ── Brand ──────────────────────────────────────────────────────
        st.markdown("### 🔍 IR System")
        st.caption("Information Retrieval")
        st.divider()

        # ── Navigation ────────────────────────────────────────────────
        current_page = st.session_state.get(SS_SELECTED_PAGE, "search")

        for page in _PAGES:
            key = page["key"]
            label = page["label"]
            icon = page["icon"]
            
            is_active = key == current_page
            
            # Create a button with key to track clicks
            if st.button(
                f"{icon} {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state[SS_SELECTED_PAGE] = key
                st.rerun()

        st.divider()

        # ── Dataset Info ───────────────────────────────────────────────
        st.markdown("**Dataset**")
        try:
            info = gateway.get_dataset_info_cached()
        except Exception:
            info = None

        if info and isinstance(info, dict):
            name = info.get("dataset_name", config.DATASET_NAME)
            doc_count = info.get("doc_count", config.DATASET_DOC_COUNT)
            st.success(f"✅ {name}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Docs", f"{doc_count:,}")
            with c2:
                q = info.get("query_count", config.DATASET_QUERY_COUNT_APPROX)
                st.metric("Queries", f"{q:,}")
        else:
            st.warning("Dataset not loaded")

        st.divider()

        # ── Index Status ───────────────────────────────────────────────
        st.markdown("**Indices**")
        try:
            idx_summary = indexing_api.get_all_index_status_cached()
        except Exception:
            idx_summary = {}

        _INDEX_ICONS = {"tfidf": "📊 TF-IDF", "bm25": "📈 BM25", "embedding": "🧠 Embedding"}
        for itype in ("tfidf", "bm25", "embedding"):
            data = idx_summary.get(itype, {}) if isinstance(idx_summary, dict) else {}
            status = str(data.get("status", "unknown")).lower() if isinstance(data, dict) else "unknown"
            ok = status in ("ready", "built", "complete", "healthy")
            status_icon = "🟢" if ok else "🔴"
            doc_count = data.get("doc_count") or data.get("document_count") or data.get("total_docs")
            count_str = f" · {doc_count:,}" if isinstance(doc_count, int) else ""
            st.caption(f"{status_icon} {_INDEX_ICONS[itype]}{count_str}")

        if st.button("🔄 Refresh indices", key="sb_refresh_idx", use_container_width=True):
            indexing_api.clear_index_status_cache()
            st.rerun()

        st.divider()

        # ── Service Health ─────────────────────────────────────────
        st.markdown("**Services**")
        try:
            health_data = health_api.check_all_services()
        except Exception:
            health_data = {}

        _SERVICE_ICONS = {
            "gateway": "🌐",
            "indexing": "🗂️",
            "search": "🔍",
            "evaluation": "📊",
            "preprocessing": "⚙️",
        }

        for svc_name, result in health_data.items():
            ok = result.get("ok") if isinstance(result, dict) else None
            ms = result.get("elapsed_ms", 0.0) if isinstance(result, dict) else 0.0
            status_icon = _STATUS_EMOJI.get(ok, "🟡")
            svc_icon = _SERVICE_ICONS.get(svc_name, "📦")
            st.caption(f"{status_icon} {svc_icon} {svc_name} · {ms:.0f}ms")

        st.divider()

        # ── Auto-refresh ────────────────────────────────────────────────
        auto = st.toggle(
            "Auto-refresh",
            value=st.session_state.get(SS_AUTO_REFRESH, False),
            key="sb_auto_refresh_toggle",
        )
        st.session_state[SS_AUTO_REFRESH] = auto

        if auto:
            interval = st.selectbox(
                "Interval (s)",
                config.AUTO_REFRESH_INTERVAL_OPTIONS,
                index=config.AUTO_REFRESH_INTERVAL_OPTIONS.index(
                    st.session_state.get(SS_AUTO_REFRESH_INTERVAL, 10)
                ),
                key="sb_refresh_interval",
            )
            st.session_state[SS_AUTO_REFRESH_INTERVAL] = interval
            last = st.session_state.get(SS_LAST_REFRESH_TIME)
            if last is None or (time.time() - last) >= interval:
                health_api.clear_cache()
                indexing_api.clear_index_status_cache()
                st.session_state[SS_LAST_REFRESH_TIME] = time.time()
                st.rerun()
