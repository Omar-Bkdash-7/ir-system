"""
Preprocessing configuration widgets — the only place preprocessing options are drawn.

Three entry points, kept apart because they are different forms on purpose:

* `render_preprocessing_config()` — the labelled form used by Search / Lab.
* `render_indexing_preprocessing_config()` — the compact form the Indexing page uses
  before a build; it returns a config only when "Apply custom preprocessing" is on.
* `render_preprocessing_preset_selector()` — quick preset picker that populates session state.

They share the session-state seeding below rather than each re-deriving defaults.
"""
from __future__ import annotations

import streamlit as st

import config
from constants import SS_PREPROCESSING_CONFIG, SS_PREPROCESSING_PRESET
from ui_style import inject_style


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


def render_indexing_preprocessing_config() -> dict | None:
    """
    Compact preprocessing form for the Indexing page.

    Returns the config dict only when the user opts into a custom configuration,
    otherwise None so the build falls back to the server defaults.
    """
    defaults = config.DEFAULT_PREPROCESSING_CONFIG

    with st.expander("Preprocessing Configuration", expanded=False):
        st.caption("Configure how documents are preprocessed before indexing")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Text Normalization**")
            normalize_unicode = st.checkbox(
                "Normalize Unicode",
                value=defaults["normalize_unicode"],
                help="Convert text to normalized Unicode form",
            )
            lowercase = st.checkbox(
                "Lowercase",
                value=defaults["lowercase"],
                help="Convert all text to lowercase",
            )
            remove_punctuation = st.checkbox(
                "Remove Punctuation",
                value=defaults["remove_punctuation"],
                help="Remove punctuation marks",
            )
            keep_numbers = st.checkbox(
                "Keep Numbers",
                value=defaults["keep_numbers"],
                help="Keep numeric tokens",
            )

        with col2:
            st.markdown("**Token Processing**")
            remove_stopwords = st.checkbox(
                "Remove Stopwords",
                value=defaults["remove_stopwords"],
                help="Remove common English stopwords",
            )
            use_lemmatization = st.checkbox(
                "Use Lemmatization",
                value=defaults["use_lemmatization"],
                help="Reduce words to base form (better, slower)",
            )
            use_stemming = st.checkbox(
                "Use Stemming",
                value=defaults["use_stemming"],
                help="Reduce words to stem (faster, less accurate)",
            )
            use_spacy_tokenizer = st.checkbox(
                "Use spaCy Tokenizer",
                value=defaults["use_spacy_tokenizer"],
                help="Use spaCy for advanced tokenization",
            )

        st.markdown("**Token Length Filters**")
        col3, col4 = st.columns(2)
        with col3:
            min_token_length = st.number_input(
                "Min Token Length",
                min_value=1,
                max_value=10,
                value=defaults["min_token_length"],
                help="Minimum character length for tokens",
            )
        with col4:
            max_token_length = st.number_input(
                "Max Token Length (0 = no limit)",
                min_value=0,
                max_value=100,
                value=0,
                help="Maximum character length for tokens",
            )

        apply_custom = st.checkbox(
            "Apply custom preprocessing",
            value=False,
            help="Use the configuration above instead of defaults",
        )

        if apply_custom:
            return {
                "normalize_unicode": normalize_unicode,
                "lowercase": lowercase,
                "remove_punctuation": remove_punctuation,
                "remove_stopwords": remove_stopwords,
                "min_token_length": min_token_length,
                "max_token_length": max_token_length if max_token_length > 0 else None,
                "use_lemmatization": use_lemmatization,
                "use_stemming": use_stemming,
                "keep_numbers": keep_numbers,
                "use_spacy_tokenizer": use_spacy_tokenizer,
                "custom_stopwords": [],
            }

    return None


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

    inject_style("preprocessing_radio")

    st.markdown(
        '<div class="ir-eyebrow" style="margin-top:10px">Morphological Processing</div>',
        unsafe_allow_html=True,
    )
    morph_options = ["Lemmatization", "Stemming", "Neither"]
    morph_key = f"{key_prefix}_morph"
    
    # Determine default only if not in session state yet
    if morph_key not in st.session_state:
        if cfg.get("use_lemmatization"):
            morph_default = 0
        elif cfg.get("use_stemming"):
            morph_default = 1
        else:
            morph_default = 2
        st.session_state[morph_key] = morph_options[morph_default]

    morph = st.radio(
        "Method",
        morph_options,
        horizontal=True,
        key=morph_key,
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
