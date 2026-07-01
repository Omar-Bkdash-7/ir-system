"""
Result card component.

render_result_card()  — single document card with rank badge, score bar, expandable text.
render_results_grid() — side-by-side columns when multiple models selected.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from constants import SS_EXPANDED_DOCS

_RANK_BADGES = {1: ("🥇", "top1"), 2: ("🥈", "top2"), 3: ("🥉", "top3")}
_SNIPPET_LEN = 420


def _score_bar(score: float) -> str:
    pct = min(max(score * 100, 0), 100)
    return (
        f'<div style="margin-top:6px;height:4px;background:var(--ctp-surface1);'
        f'border-radius:999px;overflow:hidden">'
        f'<div style="width:{pct:.1f}%;height:100%;'
        f'background:linear-gradient(90deg,var(--ctp-green),var(--ctp-teal));'
        f'border-radius:999px"></div></div>'
    )


def render_result_card(doc: Dict, rank: int, model_label: str = "") -> None:
    """Render one document result card."""
    doc_id = doc.get("doc_id", "—")
    score = float(doc.get("score", 0.0))
    text = doc.get("text", "")

    rank_icon, rank_css = _RANK_BADGES.get(rank, (f"#{rank}", ""))
    border_extra = f"border-color:var(--ctp-yellow)" if rank == 1 else (
        "border-color:var(--ctp-subtext0)" if rank == 2 else (
        "border-color:var(--ctp-peach)" if rank == 3 else ""))

    expand_key = f"expand_{model_label}_{doc_id}_{rank}"
    is_expanded = expand_key in st.session_state.get(SS_EXPANDED_DOCS, set())

    snippet = text[:_SNIPPET_LEN] + ("…" if len(text) > _SNIPPET_LEN else "")
    display_text = text if is_expanded else snippet

    st.markdown(
        f'<div class="result-card {f"result-card--{rank_css}" if rank_css else ""}" '
        f'style="{border_extra}">'
        f'<div class="result-card__header">'
        f'<span class="result-card__rank">{rank_icon} &nbsp;'
        f'<span class="result-card__docid">doc {doc_id}</span></span>'
        f'<span class="result-card__score">⭐ {score:.4f}</span>'
        f'</div>'
        f'<div class="result-card__snippet">{display_text}</div>'
        f'{_score_bar(score)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if len(text) > _SNIPPET_LEN:
        label = "▲ Show less" if is_expanded else "▼ Show more"
        if st.button(label, key=f"btn_{expand_key}", use_container_width=False):
            expanded = set(st.session_state.get(SS_EXPANDED_DOCS, set()))
            if is_expanded:
                expanded.discard(expand_key)
            else:
                expanded.add(expand_key)
            st.session_state[SS_EXPANDED_DOCS] = expanded
            st.rerun()


def render_results_grid(
    results_by_model: Dict[str, List[Dict]],
    max_display: int = 20,
) -> None:
    """
    Side-by-side columns for multi-model comparison.
    results_by_model: {"TF-IDF": [...], "BM25": [...], ...}
    """
    model_names = list(results_by_model.keys())
    if not model_names:
        st.info("No results to display.")
        return

    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        with col:
            st.markdown(f"#### {name}")
            docs = results_by_model[name][:max_display]
            if not docs:
                st.caption("No results from this model.")
            else:
                for idx, doc in enumerate(docs, 1):
                    render_result_card(doc, idx, model_label=name)


def render_refinement_banner(refinement: dict) -> None:
    """Compact inline banner shown above results when refinement was applied."""
    if not refinement or not refinement.get("applied"):
        return

    original = refinement.get("original_query", "")
    refined = refinement.get("refined_query", "")
    corrections = refinement.get("corrections", {})
    synonyms = refinement.get("synonyms_added", {})
    history_terms = refinement.get("history_terms_added", [])

    parts = []
    if corrections:
        parts.append(f"🔤 {len(corrections)} spelling fix{'es' if len(corrections) > 1 else ''}")
    if synonyms:
        total = sum(len(v) for v in synonyms.values())
        parts.append(f"📚 {total} synonym{'s' if total > 1 else ''}")
    if history_terms:
        parts.append(f"🧠 {len(history_terms)} semantic term{'s' if len(history_terms) > 1 else ''}")

    summary = " · ".join(parts) if parts else "no changes"

    with st.expander(f"✨ Query Refinement Applied — {summary}", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.info(original)
        with c2:
            st.markdown("**Refined**")
            st.success(refined)

        if corrections:
            st.markdown("**Spelling corrections**")
            rows = [{"Wrong": k, "Corrected": v} for k, v in corrections.items()]
            st.table(rows)
        if synonyms:
            st.markdown("**Synonyms added (WordNet)**")
            rows = [{"Word": k, "Synonyms": ", ".join(v)} for k, v in synonyms.items()]
            st.table(rows)
        if history_terms:
            st.markdown(f"**Semantic history terms:** `{' '.join(history_terms)}`")
            sim = refinement.get("history_similarity", 0.0)
            st.caption(f"History similarity score: {sim:.3f}")
