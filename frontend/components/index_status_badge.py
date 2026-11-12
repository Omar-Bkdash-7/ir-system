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


def render_index_card(
    index_type: str, 
    status_data: dict, 
    on_rebuild_key: str, 
    on_delete_key: str = None,
    disabled: bool = False
) -> tuple[bool, bool]:
    """
    Full card for the Index Management page: title, status, stats, Rebuild and Delete buttons.
    Returns (rebuild_clicked, delete_clicked) - True if respective button was clicked this run.
    
    Args:
        index_type: Type of index (tfidf, bm25, embedding)
        status_data: Dictionary containing index status information
        on_rebuild_key: Unique key for the rebuild button
        on_delete_key: Unique key for the delete button (optional)
        disabled: Whether the buttons should be disabled
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
        
        # Stats container with fixed height and vertical spacing
        stats_html = '<div style="min-height: 75px; display: flex; flex-direction: column; gap: 8px;">'
        
        if index_type == "tfidf":
            vocab = status_data.get("vocab_size")
            if vocab is not None:
                stats_html += f'<div class="index-card__stat">Vocab: {vocab:,}</div>'
            else:
                stats_html += '<div class="index-card__stat" style="opacity: 0.5;">Vocab: —</div>'

        elif index_type == "bm25":
            vocab = status_data.get("vocab_size")
            avg_dl = status_data.get("avg_doc_length")
            if vocab is not None:
                stats_html += f'<div class="index-card__stat">Vocab: {vocab:,}</div>'
            else:
                stats_html += '<div class="index-card__stat" style="opacity: 0.5;">Vocab: —</div>'
            if avg_dl is not None:
                stats_html += f'<div class="index-card__stat">Avg DL: {avg_dl:.1f}</div>'
            else:
                stats_html += '<div class="index-card__stat" style="opacity: 0.5;">Avg DL: —</div>'
        elif index_type == "embedding":
            shape = status_data.get("shape")
            shape_str = f"[N, {config.EMBEDDING_DIMENSIONS}]" if shape is None else str(shape)
            stats_html += f'<div class="index-card__stat">Shape: {shape_str}</div>'

        stats_html += '</div>'
        st.markdown(stats_html, unsafe_allow_html=True)

        if status_data.get("error"):
            st.caption(f"⚠️ {status_data['error']}")

        st.write("")
        
        # Buttons row
        button_help = "Upload dataset to database first" if disabled else None
        
        col1, col2 = st.columns(2)
        with col1:
            rebuild_clicked = st.button("🔨 Rebuild", key=on_rebuild_key, use_container_width=True, 
                                       disabled=disabled, help=button_help)
        with col2:
            delete_clicked = False
            if on_delete_key:
                # Only show delete button if index exists (status is ready)
                delete_disabled = disabled or status != "ready"
                delete_help = "No index to delete" if status != "ready" else "Delete this index"
                delete_clicked = st.button("🗑️ Delete", key=on_delete_key, use_container_width=True,
                                         disabled=delete_disabled, help=delete_help, type="secondary")
        
        return rebuild_clicked, delete_clicked