"""
User Guide Page - Bilingual (EN/AR)

The copy itself lives in `content/guide_en.py` and `content/guide_ar.py` as an
ordered list of blocks; this module only draws them (and switches language), so
editing the guide means editing text, not Python control flow.
"""
from __future__ import annotations

import streamlit as st

from content import guide_ar, guide_en


def render() -> None:
    st.markdown("## 📚 User Guide / دليل المستخدم")

    col1, col2, _spacer = st.columns([1, 1, 8])
    with col1:
        if st.button("English", use_container_width=True):
            st.session_state["guide_lang"] = "en"
            st.rerun()
    with col2:
        if st.button("العربية", use_container_width=True):
            st.session_state["guide_lang"] = "ar"
            st.rerun()

    if st.session_state.get("guide_lang", "en") == "ar":
        _render_blocks(guide_ar.BLOCKS)
    else:
        _render_blocks(guide_en.BLOCKS)


def _render_blocks(blocks) -> None:
    """Draw one language's blocks in order: text, rules and collapsible sections."""
    for kind, payload in blocks:
        if kind == "divider":
            st.markdown("---")
        elif kind == "html":
            st.markdown(payload, unsafe_allow_html=True)
        elif kind == "expander":
            with st.expander(payload["title"], expanded=payload["expanded"]):
                st.markdown(payload["body"])
        else:
            st.markdown(payload)
