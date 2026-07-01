"""Pure-rendering components for index readiness: a compact sidebar badge and a full status card."""
from __future__ import annotations

import streamlit as st

import config

_ICONS = {"tfidf": "📊", "bm25": "📈", "embedding": "🧠"}
_LABELS = config.RETRIEVAL_MODEL_LABELS


def _status_of(status_data: dict) -> str:
    """Normalize whatever the backend calls it into: ready | building | not_built | unknown."""
    if not status_data:
        return "unknown"
    raw = str(status_data.get("status", "unknown")).lower()
    if raw in ("ready", "built", "complete", "completed", "healthy"):
        return "ready"
    if raw in ("building", "in_progress", "indexing"):
        return "building"
    if raw in ("not_built", "not built", "missing", "none"):
        return "not_built"
    return "unknown"


def _dot_class(status: str) -> str:
    return {"ready": "ir-dot--green", "building": "ir-dot--yellow", "not_built": "ir-dot--red"}.get(
        status, "ir-dot--yellow"
    )


def _status_text(status: str) -> str:
    return {"ready": "READY", "building": "BUILDING…", "not_built": "NOT BUILT"}.get(status, "UNKNOWN")


def render_index_badge_compact(index_type: str, status_data: dict) -> None:
    """Small one-line badge used in the sidebar."""
    status = _status_of(status_data)
    doc_count = status_data.get("doc_count") or status_data.get("document_count")
    count_str = f" · {doc_count:,}" if isinstance(doc_count, (int, float)) else ""
    st.markdown(
        f'<div class="ir-badge"><span class="ir-dot {_dot_class(status)}"></span>'
        f'{_ICONS.get(index_type, "🔹")} {_LABELS.get(index_type, index_type)} '
        f'<span class="ir-mono" style="color:var(--ctp-subtext0)">{_status_text(status)}{count_str}</span></div>',
        unsafe_allow_html=True,
    )


def render_index_card(index_type: str, status_data: dict, on_rebuild_key: str) -> bool:
    """
    Full card for the Index Management page: title, status, stats, Rebuild button.
    Returns True if the Rebuild button was clicked this run.
    """
    status = _status_of(status_data)
    with st.container(border=True):
        st.markdown(
            f'<div class="index-card"><div class="index-card__title">'
            f'{_ICONS.get(index_type, "🔹")} {_LABELS.get(index_type, index_type)}</div>'
            f'<div class="ir-badge" style="justify-content:center"><span class="ir-dot {_dot_class(status)}">'
            f'</span>{_status_text(status)}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        doc_count = status_data.get("doc_count") or status_data.get("document_count")
        if doc_count is not None:
            st.markdown(f'<div class="index-card__stat">{doc_count:,} docs</div>', unsafe_allow_html=True)

        if index_type == "tfidf":
            vocab = status_data.get("vocab_size")
            size_mb = status_data.get("size_mb")
            if vocab is not None:
                st.markdown(f'<div class="index-card__stat">Vocab: {vocab:,}</div>', unsafe_allow_html=True)
            if size_mb is not None:
                st.markdown(f'<div class="index-card__stat">Size: {size_mb} MB</div>', unsafe_allow_html=True)
        elif index_type == "bm25":
            vocab = status_data.get("vocab_size")
            avg_dl = status_data.get("avg_doc_length")
            if vocab is not None:
                st.markdown(f'<div class="index-card__stat">Vocab: {vocab:,}</div>', unsafe_allow_html=True)
            if avg_dl is not None:
                st.markdown(f'<div class="index-card__stat">Avg DL: {avg_dl}</div>', unsafe_allow_html=True)
        elif index_type == "embedding":
            shape = status_data.get("shape")
            size_mb = status_data.get("size_mb")
            shape_str = f"[N, {config.EMBEDDING_DIMENSIONS}]" if shape is None else str(shape)
            st.markdown(f'<div class="index-card__stat">Shape: {shape_str}</div>', unsafe_allow_html=True)
            if size_mb is not None:
                st.markdown(f'<div class="index-card__stat">Size: {size_mb} MB</div>', unsafe_allow_html=True)

        if status_data.get("error"):
            st.caption(f"⚠️ {status_data['error']}")

        st.write("")
        return st.button("🔨 Rebuild", key=on_rebuild_key, use_container_width=True)