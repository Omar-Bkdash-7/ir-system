"""
IR System — Streamlit entry point.

Run with: streamlit run frontend/app_new.py
(The original frontend/app.py is preserved as a backup.)

Architecture:
  app_new.py  ← this file: page config, CSS, session state init, router
  config.py   ← all URLs, constants, defaults
  constants.py ← session state keys + DEFAULTS dict
  api/        ← one module per microservice
  components/ ← reusable UI building blocks
  pages/      ← one module per page; each exports render()
  styles/     ← theme.css (Catppuccin Mocha)
"""
from __future__ import annotations

import sys
import os

# Ensure frontend/ is on the import path when running as
# `streamlit run frontend/app_new.py` from the project root.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import streamlit as st

# ── Page config (must be the very first Streamlit call) ────────────────────
st.set_page_config(
    page_title="IR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports that depend on sys.path being set up correctly ─────────────────
import config                                               # noqa: E402
from constants import DEFAULTS, SS_SELECTED_PAGE           # noqa: E402
from components.sidebar import render_sidebar              # noqa: E402
import pages.search as page_search                         # noqa: E402
import pages.dataset as page_dataset                       # noqa: E402
import pages.indexing as page_indexing                     # noqa: E402
import pages.preprocessing as page_preprocessing          # noqa: E402
import pages.evaluation as page_evaluation                 # noqa: E402
import pages.query_refinement as page_qr                  # noqa: E402
import pages.system_status as page_sysmon                 # noqa: E402


# ── Load CSS ───────────────────────────────────────────────────────────────
def _load_css() -> None:
    css_path = os.path.join(_here, "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as fh:
            css = fh.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_load_css()


# ── Session state initialisation ──────────────────────────────────────────
def _init_session_state() -> None:
    """Seed every key with its default value on the first run."""
    for key, default in DEFAULTS.items():
        st.session_state.setdefault(key, default)

    # Generate a stable session ID (used by query refinement service)
    if not st.session_state.get("session_id"):
        import uuid
        st.session_state["session_id"] = str(uuid.uuid4())[:8]


_init_session_state()


# ── Router ─────────────────────────────────────────────────────────────────
_PAGE_MAP = {
    "search":           page_search,
    "dataset":          page_dataset,
    "indexing":         page_indexing,
    "preprocessing":    page_preprocessing,
    "evaluation":       page_evaluation,
    "query_refinement": page_qr,
    "system_status":    page_sysmon,
}


def _render_page(page_key: str) -> None:
    module = _PAGE_MAP.get(page_key)
    if module is None:
        st.error(f"Unknown page: {page_key!r}")
        return
    module.render()


# ── Layout ─────────────────────────────────────────────────────────────────
current_page = render_sidebar()          # sidebar draws itself and returns active page key
_render_page(current_page)
