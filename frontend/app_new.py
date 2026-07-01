"""
IR System — Streamlit entry point.
Run with: streamlit run frontend/app_new.py
"""
from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import streamlit as st

st.set_page_config(
    page_title="IR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config                                        # noqa: E402
from constants import DEFAULTS, SS_SELECTED_PAGE    # noqa: E402
from components.sidebar import render_sidebar       # noqa: E402
import pages.search as page_search                  # noqa: E402
import pages.dataset as page_dataset               # noqa: E402
import pages.indexing as page_indexing             # noqa: E402
import pages.preprocessing as page_preprocessing   # noqa: E402
import pages.evaluation as page_evaluation         # noqa: E402
import pages.query_refinement as page_qr           # noqa: E402
import pages.system_status as page_sysmon          # noqa: E402


# ── CSS ──────────────────────────────────────────────────────────────────
def _load_css() -> None:
    css_path = os.path.join(_here, "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as fh:
            css = fh.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_load_css()


# ── Session state init ────────────────────────────────────────────────────
def _init_session_state() -> None:
    for key, default in DEFAULTS.items():
        st.session_state.setdefault(key, default)
    if not st.session_state.get("session_id"):
        import uuid
        st.session_state["session_id"] = str(uuid.uuid4())[:8]


_init_session_state()


# ── Page map ──────────────────────────────────────────────────────────────
_PAGE_MAP = {
    "search":           page_search,
    "dataset":          page_dataset,
    "indexing":         page_indexing,
    "preprocessing":    page_preprocessing,
    "evaluation":       page_evaluation,
    "query_refinement": page_qr,
    "system_status":    page_sysmon,
}


# ── Render ────────────────────────────────────────────────────────────────
# 1. Draw the sidebar (sets session_state[SS_SELECTED_PAGE] on click)
render_sidebar()

# 2. Read the active page key — never use the return value of render_sidebar()
active_page = st.session_state.get(SS_SELECTED_PAGE, "search")

# 3. Render the active page
module = _PAGE_MAP.get(active_page)
if module is not None:
    module.render()
else:
    st.error(f"Unknown page: {active_page!r}")
