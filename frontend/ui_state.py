"""
State shared by more than one page.

Streamlit throws away a widget's value as soon as the widget is not rendered,
so a dataset picked on one page vanished the moment the user navigated
somewhere else. The canonical selection therefore lives in session state
(`SS_SELECTED_DATASET`) and every page's selector reads it as its default.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import streamlit as st

from constants import SS_SELECTED_DATASET


def get_selected_dataset() -> Optional[str]:
    """The dataset the user is working on, whichever page they picked it from."""
    return st.session_state.get(SS_SELECTED_DATASET)


def set_selected_dataset(dataset: Optional[str]) -> None:
    st.session_state[SS_SELECTED_DATASET] = dataset


def dataset_selector(
    options: Sequence[str],
    *,
    key: str,
    label: str = "Select Dataset",
    help: Optional[str] = None,
    placeholder: str = "Choose a dataset…",
    label_visibility: str = "visible",
    format_func: Optional[Callable[[str], str]] = None,
    on_change: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """
    Dataset picker bound to the shared selection.

    The chosen value is remembered for the whole session, so switching pages
    keeps the working dataset instead of resetting to an empty selector. When
    the remembered dataset is not in `options` the selector falls back to its
    placeholder, which is what the pages showed before anything was picked.
    """
    option_list = list(options)
    current = get_selected_dataset()
    default_index = option_list.index(current) if current in option_list else None

    def _remember() -> None:
        chosen = st.session_state.get(key)
        if chosen:
            set_selected_dataset(chosen)
        if on_change is not None:
            on_change()

    extra = {"format_func": format_func} if format_func is not None else {}
    selected = st.selectbox(
        label,
        option_list,
        index=default_index,
        key=key,
        placeholder=placeholder,
        help=help,
        label_visibility=label_visibility,
        on_change=_remember,
        **extra,
    )

    if selected:
        set_selected_dataset(selected)
    return selected
