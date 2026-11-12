"""
Preprocessing Lab page — interactive config + live token preview.
"""
from __future__ import annotations

import streamlit as st

from api.preprocessing import preprocess
from components.preprocessing_config import (
    render_preprocessing_config,
    render_preprocessing_preset_selector,
)
from constants import (
    SS_PREPROCESSING_LAB_INPUT,
    SS_PREPROCESSING_LAB_RESULT,
)


def _render_token_chips(tokens: list[str]) -> None:
    chips = "".join(f'<span class="chip">{t}</span>' for t in tokens)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def render() -> None:
    st.markdown("## ⚗️ Preprocessing Lab")
    st.caption(
        "Configure the NLP preprocessing pipeline and see token output in real time. "
        "The same config is used during index building and query processing."
    )

    # ── Preset selector ────────────────────────────────────────────────────
    st.markdown("### Quick Preset")
    render_preprocessing_preset_selector()

    st.divider()
    st.markdown("### Pipeline Configuration")
    cfg = render_preprocessing_config(key_prefix="lab_prep")

    st.divider()
    st.markdown("### Live Token Preview")

    test_input = st.text_area(
        "Test text",
        value=st.session_state.get(
            SS_PREPROCESSING_LAB_INPUT,
            "What is the best way to learn machine learning and deep learning?",
        ),
        height=90,
        key="lab_text_input",
    )
    st.session_state[SS_PREPROCESSING_LAB_INPUT] = test_input

    if st.button("🔬 Run Preprocessing", type="primary", key="btn_run_prep"):
        if test_input.strip():
            with st.spinner("Calling preprocessing service…"):
                resp = preprocess(test_input, cfg)
                if resp.ok and isinstance(resp.data, dict):
                    st.session_state[SS_PREPROCESSING_LAB_RESULT] = resp.data
                else:
                    st.error(f"Preprocessing service error: {resp.error}")
                    st.session_state[SS_PREPROCESSING_LAB_RESULT] = None
        else:
            st.warning("Enter some text first.")

    result = st.session_state.get(SS_PREPROCESSING_LAB_RESULT)
    if result:
        tokens: list = result.get("tokens", [])
        original: str = result.get("original", test_input)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Original words", len(original.split()))
        with c2:
            st.metric("Tokens after preprocessing", len(tokens))
        with c3:
            removed = len(original.split()) - len(tokens)
            st.metric("Removed", removed)

        st.markdown("**Resulting tokens:**")
        if tokens:
            _render_token_chips(tokens)
        else:
            st.warning("No tokens produced — check your configuration (min_token_length, stopwords).")

        with st.expander("Raw JSON response", expanded=False):
            st.json(result)

    st.divider()
    st.markdown("### Configuration Reference")
    st.markdown("""
| Option | Default | Effect |
|---|---|---|
| `use_spacy_tokenizer` | `True` | Uses spaCy; faster than NLTK, enables lemmatization via spaCy |
| `normalize_unicode` | `True` | Converts accented chars, strips non-ASCII |
| `lowercase` | `True` | Folds case |
| `remove_punctuation` | `True` | Strips all punctuation tokens |
| `remove_stopwords` | `True` | Removes NLTK English stopwords |
| `keep_numbers` | `False` | If False, pure numeric tokens are dropped |
| `min_token_length` | `2` | Drops tokens shorter than N chars |
| `max_token_length` | `None` | Drops tokens longer than N chars (disabled by default) |
| `use_lemmatization` | `True` | Reduces to dictionary form (spaCy or WordNet) |
| `use_stemming` | `False` | Porter stemmer — mutually exclusive with lemmatization |
| `custom_stopwords` | `[]` | Extra words to remove on top of NLTK defaults |
""")
