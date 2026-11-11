"""
Loader for the CSS that Streamlit has to receive as inline `<style>` tags.

The bulk of the styling lives in `styles/theme.css` and is injected once at
startup. A few rules have to be injected at a precise point of a page (they
override Streamlit widgets or depend on the widget being present), so they live
in their own files under `styles/` and are loaded from there instead of being
inlined as string literals in Python.

Keeping them in files means a stylesheet can be edited as CSS (syntax
highlighting, no escaping of `{` in f-strings) while the injection order stays
exactly what it was.
"""
from __future__ import annotations

import os

import streamlit as st

_STYLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles")

# name -> (mtime, css). Keyed by modification time so editing a stylesheet
# shows up on the next rerun instead of needing a process restart.
_CACHE: dict[str, tuple[float, str]] = {}


def stylesheet_path(name: str) -> str:
    """Absolute path of a stylesheet, e.g. ``stylesheet_path("sidebar_nav")``."""
    return os.path.join(_STYLES_DIR, *name.split("/")) + ".css"


def load_css(name: str) -> str:
    """Read one stylesheet from `frontend/styles/` (`name` may contain a subdirectory)."""
    path = stylesheet_path(name)
    mtime = os.path.getmtime(path)
    cached = _CACHE.get(name)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with open(path, encoding="utf-8") as fh:
        css = fh.read()
    _CACHE[name] = (mtime, css)
    return css


def clear_css_cache() -> None:
    """Drop every cached stylesheet (used by tests)."""
    _CACHE.clear()


def inject_style(name: str, *, use_html: bool = False) -> None:
    """
    Inject a stylesheet into the current page.

    `use_html=True` renders through `st.html` (what the app shell and the theme
    use); the default renders through `st.markdown`, which is where the
    page-level overrides used to live. Both paths produce the same
    `<style>` element they produced before the CSS was moved out of Python.
    """
    css = load_css(name)
    if use_html:
        st.html(f"<style>{css}</style>")
    else:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
