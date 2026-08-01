"""
BM25 parameter widget — reusable k1 / b sliders with academic explanations.
Returns (k1, b) tuple.
"""
from __future__ import annotations

import streamlit as st

import config


def render_bm25_params(key_prefix: str = "bm25") -> tuple[float, float]:
    """
    Render k1 and b sliders with explanations.
    Returns (k1, b).
    key_prefix must be unique per usage site to avoid widget key collisions.
    """
    st.markdown(
        '<div class="ir-eyebrow">BM25 Parameters</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        k1 = st.slider(
            "k1 — Term Saturation",
            min_value=config.MIN_BM25_K1,
            max_value=config.MAX_BM25_K1,
            value=config.DEFAULT_BM25_K1,
            step=0.1,
            key=f"{key_prefix}_k1",
            help=(
                "Controls how quickly term-frequency contribution saturates. "
                "Higher k1 → more reward for repeated query terms. "
                "Typical range: 1.2 – 2.0"
            ),
        )
    with c2:
        b = st.slider(
            "b — Length Normalization",
            min_value=config.MIN_BM25_B,
            max_value=config.MAX_BM25_B,
            value=config.DEFAULT_BM25_B,
            step=0.05,
            key=f"{key_prefix}_b",
            help=(
                "Controls document-length penalization. "
                "b=1.0 → full normalization, b=0.0 → no normalization. "
                "Default 0.75 works well for most corpora."
            ),
        )
    st.caption(
        f"Current: **k1 = {k1:.1f}**, **b = {b:.2f}**"
    )
    return k1, b
