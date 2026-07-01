"""
Preprocessing configuration widget.

render_preprocessing_config() — full config form, returns dict matching PreprocessingConfig.
render_preprocessing_preset_selector() — quick preset picker that populates session state.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

import config
from constants import SS_PREPROCESSING_CONFIG, SS_PREPROCESSING_PRESET


def _get_current_config() -> dict:
    """Return the active config from session state, seeding defaults if absent."""
    if st.session_state.get(SS_PREPROCESSING_CONFIG) is None:
        st.session_state[SS_PREPROCESSING_CONFIG] = dict(config.DEFAULT_PREPROCESSING_CONFIG)
    return st.session_state[SS_PREPROCESSING_CONFIG]


def render_preprocessing_preset_selector() -> None:
    """Compact preset picker row. Updates session state directly."""
    preset_names = list(config.PREPROCESSING_PRESETS.keys())
    current = st.session_state.get(SS_PREPROCESSING_PRESET, "Default (Academic)")
    chosen = st.selectbox(
        "Quick preset",
        preset_names,
        index=preset_names.index(current) if current in preset_names else 0,
        key="prep_preset_selector",
    )
    if chosen != current or st.session_state.get(SS_PREPROCESSING_CONFIG) is None:
        st.session_state[SS_PREPROCESSING_PRESET] = chosen
        st.session_state[SS_PREPROCESSING_CONFIG] = dict(config.PREPROCESSING_PRESETS[chosen])


def render_preprocessing_config(key_prefix: str = "prep") -> dict:
    """
    Full preprocessing config form.
    Returns the config dict (also stored in session state).
    key_prefix must be unique per usage site.
    """
    cfg = _get_current_config()

    st.markdown(
        '<div class="ir-eyebrow">Tokenizer</div>', unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    with c1:
        cfg["use_spacy_tokenizer"] = st.checkbox(
            "Use spaCy tokenizer",
            value=cfg.get("use_spacy_tokenizer", True),
            key=f"{key_prefix}_spacy",
            help="Uses spaCy for tokenization + lemmatization. Disable for faster SimpleTokenizer.",
        )
    with c2:
        cfg["normalize_unicode"] = st.checkbox(
            "Normalize Unicode",
            value=cfg.get("normalize_unicode", True),
            key=f"{key_prefix}_unicode",
        )

    st.markdown(
        '<div class="ir-eyebrow" style="margin-top:10px">Filtering</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cfg["lowercase"] = st.checkbox(
            "Lowercase",
            value=cfg.get("lowercase", True),
            key=f"{key_prefix}_lower",
        )
    with c2:
        cfg["remove_punctuation"] = st.checkbox(
            "Remove punctuation",
            value=cfg.get("remove_punctuation", True),
            key=f"{key_prefix}_punct",
        )
    with c3:
        cfg["remove_stopwords"] = st.checkbox(
            "Remove stopwords",
            value=cfg.get("remove_stopwords", True),
            key=f"{key_prefix}_stop",
        )
    with c4:
        cfg["keep_numbers"] = st.checkbox(
            "Keep numbers",
            value=cfg.get("keep_numbers", False),
            key=f"{key_prefix}_nums",
        )

    st.markdown(
        '<div class="ir-eyebrow" style="margin-top:10px">Token Length</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        cfg["min_token_length"] = st.number_input(
            "Min token length",
            min_value=config.MIN_TOKEN_LENGTH_RANGE[0],
            max_value=config.MIN_TOKEN_LENGTH_RANGE[1],
            value=int(cfg.get("min_token_length", 2)),
            step=1,
            key=f"{key_prefix}_minlen",
        )
    with c2:
        use_max = st.checkbox(
            "Set max token length",
            value=cfg.get("max_token_length") is not None,
            key=f"{key_prefix}_use_max",
        )
        if use_max:
            cfg["max_token_length"] = st.number_input(
                "Max token length",
                min_value=3,
                max_value=100,
                value=int(cfg.get("max_token_length") or 30),
                step=1,
                key=f"{key_prefix}_maxlen",
            )
        else:
            cfg["max_token_length"] = None

    st.markdown(
        '<div class="ir-eyebrow" style="margin-top:10px">Morphological Processing</div>',
        unsafe_allow_html=True,
    )
    morph_options = ["Lemmatization", "Stemming", "Neither"]
    if cfg.get("use_lemmatization"):
        morph_default = 0
    elif cfg.get("use_stemming"):
        morph_default = 1
    else:
        morph_default = 2

    morph = st.radio(
        "Method",
        morph_options,
        index=morph_default,
        horizontal=True,
        key=f"{key_prefix}_morph",
        help=(
            "Lemmatization (spaCy/WordNet): slower, more accurate. "
            "Stemming (Porter): faster, more aggressive."
        ),
    )
    cfg["use_lemmatization"] = morph == "Lemmatization"
    cfg["use_stemming"] = morph == "Stemming"

    st.markdown(
        '<div class="ir-eyebrow" style="margin-top:10px">Custom Stopwords</div>',
        unsafe_allow_html=True,
    )
    custom_raw = st.text_input(
        "Additional stopwords (comma-separated)",
        value=", ".join(cfg.get("custom_stopwords") or []),
        key=f"{key_prefix}_custom_stop",
        placeholder="e.g. hello, world, foo",
    )
    cfg["custom_stopwords"] = (
        [w.strip() for w in custom_raw.split(",") if w.strip()]
        if custom_raw.strip()
        else []
    )

    st.session_state[SS_PREPROCESSING_CONFIG] = cfg
    return dict(cfg)
