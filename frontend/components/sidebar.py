"""
Sidebar component — always visible.
Renders: navigation, service health badges, index status, dataset info.
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
    SS_SERVICE_HEALTH,
)

_PAGES = [
    ("search",           "🔍 Search"),
    ("dataset",          "📊 Dataset & Preprocessing"),
    ("indexing",         "🗂️ Index Management"),
    ("evaluation",       "📈 Evaluation"),
    ("query_refinement", "🔬 Query Refinement Lab"),
    ("system_status",    "⚙️ System Monitor"),
]


def _health_dot(ok: Optional[bool]) -> str:
    if ok is True:
        return '<span class="ir-dot ir-dot--green"></span>'
    if ok is False:
        return '<span class="ir-dot ir-dot--red"></span>'
    return '<span class="ir-dot ir-dot--yellow"></span>'


def render_sidebar() -> str:
    """
    Renders the full sidebar.
    Returns the currently selected page key (str).
    """
    with st.sidebar:
        # ── Brand ────────────────────────────────────────────────────────
        st.markdown(
            '<h1 style="font-size:1.3rem;margin-bottom:0">🔍 IR System</h1>'
            '<p style="color:var(--ctp-subtext0);font-size:0.8rem;margin-top:2px">'
            'Information Retrieval · beir/quora</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Navigation ────────────────────────────────────────────────────
        st.markdown('<div class="ir-eyebrow">Navigation</div>', unsafe_allow_html=True)
        current_page = st.session_state.get(SS_SELECTED_PAGE, "search")
        for key, label in _PAGES:
            is_active = key == current_page
            style = (
                "background:var(--ctp-blue);color:var(--ctp-crust);"
                "border-radius:6px;padding:6px 12px;margin:2px 0;"
                "font-weight:700;cursor:default"
                if is_active
                else "padding:6px 12px;margin:2px 0;color:var(--ctp-text);"
                "cursor:pointer;border-radius:6px"
            )
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state[SS_SELECTED_PAGE] = key
                st.rerun()

        st.divider()

        # ── Dataset Info ──────────────────────────────────────────────────
        st.markdown('<div class="ir-eyebrow">Dataset</div>', unsafe_allow_html=True)
        info = gateway.get_dataset_info_cached()
        if info:
            name = info.get("dataset_name", config.DATASET_NAME)
            doc_count = info.get("doc_count", config.DATASET_DOC_COUNT)
            st.markdown(
                f'<div class="ir-badge"><span class="ir-dot ir-dot--green"></span>'
                f'<span style="color:var(--ctp-green)">{name}</span></div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Docs", f"{doc_count:,}")
            with c2:
                q = info.get("query_count", config.DATASET_QUERY_COUNT_APPROX)
                st.metric("Queries", f"{q:,}")
        else:
            st.markdown(
                '<div class="ir-badge"><span class="ir-dot ir-dot--red"></span>'
                '<span style="color:var(--ctp-subtext0)">Not loaded</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Index Status ──────────────────────────────────────────────────
        st.markdown('<div class="ir-eyebrow">Indices</div>', unsafe_allow_html=True)
        idx_summary = indexing_api.get_all_index_status_cached()
        from components.index_status_badge import render_index_badge_compact
        for itype in ("tfidf", "bm25", "embedding"):
            render_index_badge_compact(itype, idx_summary.get(itype, {}))

        if st.button("🔄 Refresh indices", key="sb_refresh_idx", use_container_width=True):
            indexing_api.clear_index_status_cache()
            st.rerun()

        st.divider()

        # ── Service Health (compact) ──────────────────────────────────────
        st.markdown('<div class="ir-eyebrow">Services</div>', unsafe_allow_html=True)
        health_data = health_api.check_all_services()
        for svc_name, result in health_data.items():
            dot = _health_dot(result.get("ok"))
            ms = result.get("elapsed_ms", 0.0)
            st.markdown(
                f'<div class="ir-badge">{dot}'
                f'<span style="font-size:0.82em">{svc_name}</span>'
                f'<span class="ir-mono" style="color:var(--ctp-subtext0);font-size:0.78em">'
                f'{ms:.0f}ms</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Auto-refresh toggle ───────────────────────────────────────────
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

        return st.session_state.get(SS_SELECTED_PAGE, "search")
