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


# ── CSS ──────────────────────────────────────────────────────────────────
def _load_css() -> None:
    # Inject Google Fonts via link tag (avoids @import parsing issue in st.html)
    st.html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
        '&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )
    
    # Additional inline CSS to hide multipage navigation and remove top spacing
    st.html("""
    <style>
    /* CRITICAL: Hide Streamlit's auto-generated multipage navigation */
    [data-testid="stSidebarNav"],
    nav[aria-label="Page navigation"],
    ul[data-testid="stSidebarNav"],
    section[data-testid="stSidebar"] nav,
    div[data-testid="stSidebarNav"],
    .css-1dp5vir,
    .css-17lntkn {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Remove all top spacing from main content */
    .main .block-container,
    [data-testid="block-container"],
    .main > div:first-child {
        padding-top: 0.5rem !important;
        margin-top: 0 !important;
    }
    
    /* Remove all top spacing from sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
        margin-top: 0 !important;
    }
    
    /* Force header to have zero height */
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        display: none !important;
    }
    </style>
    """)
    
    # Inject theme CSS
    css_path = os.path.join(_here, "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as fh:
            css = fh.read()
        st.html(f"<style>{css}</style>")


_load_css()


# ── Session state init ────────────────────────────────────────────────────
def _init_session_state() -> None:
    for key, default in DEFAULTS.items():
        st.session_state.setdefault(key, default)
    if not st.session_state.get("session_id"):
        import uuid
        st.session_state["session_id"] = str(uuid.uuid4())[:8]


_init_session_state()


# ── Page map with lazy loading ────────────────────────────────────────────
def _get_page_module(page_key: str):
    """Lazy load page modules only when needed."""
    if page_key == "guide":
        import app_pages.guide as page
    elif page_key == "dataset":
        import app_pages.dataset as page
    elif page_key == "search":
        import app_pages.search as page
    elif page_key == "indexing":
        import app_pages.indexing as page
    elif page_key == "preprocessing":
        import app_pages.preprocessing as page
    elif page_key == "evaluation":
        import app_pages.evaluation as page
    elif page_key == "query_refinement":
        import app_pages.query_refinement as page
    elif page_key == "clustering":
        import app_pages.clustering as page
    elif page_key == "rag":
        import app_pages.rag as page
    elif page_key == "system_status":
        import app_pages.system_status as page
    else:
        return None
    return page


# ── Render ────────────────────────────────────────────────────────────────
# 1. Draw the sidebar (sets session_state[SS_SELECTED_PAGE] on click)
render_sidebar()

# 2. Read the active page key
active_page = st.session_state.get(SS_SELECTED_PAGE, "guide")

# 3. Lazy load and render the active page
module = _get_page_module(active_page)
if module is not None:
    module.render()
else:
    st.error(f"Unknown page: {active_page!r}")
