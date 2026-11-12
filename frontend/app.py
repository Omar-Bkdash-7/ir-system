"""
IR System — Streamlit entry point.

Run with: streamlit run frontend/app.py

`frontend/` is not a package (Streamlit executes this file directly), so the
directory is put on `sys.path` to make `config`, `api.*`, `components.*` and
`app_pages.*` importable regardless of the working directory.
"""
from __future__ import annotations

import os
import sys
import uuid

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

from constants import DEFAULTS, SS_SELECTED_PAGE    # noqa: E402
from obs import get_logger, setup_logging           # noqa: E402
from components.sidebar import render_sidebar       # noqa: E402
from ui_style import inject_style                    # noqa: E402


# ── CSS ──────────────────────────────────────────────────────────────────
def _load_css() -> None:
    # Google Fonts via link tag (avoids the @import parsing issue in st.html)
    st.html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
        '&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )

    # Shell overrides (hidden multipage nav, no top spacing), then the theme
    inject_style("app_shell", use_html=True)
    inject_style("theme", use_html=True)


def _init_session_state() -> None:
    for key, default in DEFAULTS.items():
        st.session_state.setdefault(key, default)
    if not st.session_state.get("session_id"):
        st.session_state["session_id"] = str(uuid.uuid4())[:8]


setup_logging()
_log = get_logger("app")

_load_css()


# ── Session state init ────────────────────────────────────────────────────
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
if st.session_state.get("_last_rendered_page") != active_page:
    st.session_state["_last_rendered_page"] = active_page
    _log.info("page -> %s (session %s)", active_page, st.session_state.get("session_id"))

module = _get_page_module(active_page)
if module is not None:
    module.render()
else:
    st.error(f"Unknown page: {active_page!r}")
