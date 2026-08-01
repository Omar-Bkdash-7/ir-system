"""
Query Refinement Lab — demo and config page.
"""
from __future__ import annotations

from typing import List

import streamlit as st

import config
from api.gateway import refine_query
from constants import (
    SS_REFINEMENT_LAB_CONFIG,
    SS_REFINEMENT_LAB_HISTORY_INPUT,
    SS_REFINEMENT_LAB_QUERY_INPUT,
    SS_REFINEMENT_LAB_RESULT,
    SS_REFINEMENT_HISTORY,
    SS_SESSION_ID,
)


def _diff_html(original: str, refined: str) -> str:
    """Simple word-level diff highlighting."""
    orig_words = original.lower().split()
    ref_words = refined.lower().split()
    orig_set = set(orig_words)
    html_parts = []
    for word in ref_words:
        if word in orig_set:
            html_parts.append(f'<span class="diff-unchanged">{word}</span>')
        else:
            html_parts.append(f'<span class="diff-added">{word}</span>')
    for word in orig_words:
        if word not in set(ref_words):
            pass  # removed words shown in original section
    return " ".join(html_parts)


def render() -> None:
    st.markdown("## 🔬 Query Refinement Lab")
    st.caption(
        "Inspect and tune every stage of the query refinement pipeline: "
        "spelling correction, WordNet synonym expansion, and semantic history-based boosting."
    )

    # ── Configuration ──────────────────────────────────────────────────────
    with st.expander("⚙️ Refinement Configuration", expanded=False):
        st.caption(
            "These values are informational — they reflect the server-side defaults "
            "from `configs/config.yaml`. The server does not currently accept per-request "
            "config overrides; this panel documents the active configuration."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Max edit distance",
                      *config.REFINEMENT_MAX_EDIT_DISTANCE_RANGE,
                      config.REFINEMENT_MAX_EDIT_DISTANCE_DEFAULT,
                      key="rl_edit_dist", disabled=True)
            st.slider("Confidence threshold",
                      *config.REFINEMENT_CONFIDENCE_THRESHOLD_RANGE,
                      config.REFINEMENT_CONFIDENCE_THRESHOLD_DEFAULT,
                      step=0.05, key="rl_conf", disabled=True)
            st.slider("Max synonyms per word",
                      *config.REFINEMENT_MAX_SYNONYMS_RANGE,
                      config.REFINEMENT_MAX_SYNONYMS_DEFAULT,
                      key="rl_syns", disabled=True)
        with c2:
            st.slider("Max WordNet synsets",
                      *config.REFINEMENT_MAX_SYNSETS_RANGE,
                      config.REFINEMENT_MAX_SYNSETS_DEFAULT,
                      key="rl_synsets", disabled=True)
            st.slider("Min term length for expansion",
                      *config.REFINEMENT_MIN_TERM_LENGTH_RANGE,
                      config.REFINEMENT_MIN_TERM_LENGTH_DEFAULT,
                      key="rl_minlen", disabled=True)
            st.slider("Similarity threshold (history)",
                      *config.REFINEMENT_HISTORY_SIMILARITY_THRESHOLD_RANGE,
                      config.REFINEMENT_HISTORY_SIMILARITY_THRESHOLD_DEFAULT,
                      step=0.05, key="rl_sim", disabled=True)

    st.divider()

    # ── Live demo ──────────────────────────────────────────────────────────
    st.markdown("### Live Refinement Demo")

    query_input = st.text_input(
        "Input query",
        value=st.session_state.get(SS_REFINEMENT_LAB_QUERY_INPUT, "informtion retrival system"),
        placeholder="e.g. informtion retrival sistem",
        key="rl_query",
    )
    st.session_state[SS_REFINEMENT_LAB_QUERY_INPUT] = query_input

    history_raw = st.text_area(
        "History queries (one per line)",
        value=st.session_state.get(SS_REFINEMENT_LAB_HISTORY_INPUT, ""),
        height=80,
        key="rl_history",
        placeholder="document similarity\ninformation extraction\n...",
    )
    st.session_state[SS_REFINEMENT_LAB_HISTORY_INPUT] = history_raw
    history: List[str] = [l.strip() for l in history_raw.splitlines() if l.strip()]

    if st.button("🔬 Analyze Refinement", type="primary", key="btn_rl_analyze"):
        if not query_input.strip():
            st.error("Enter a query.")
        else:
            with st.spinner("Calling refinement service…"):
                resp = refine_query(
                    query_input,
                    history,
                    enabled=True,
                    session_id=st.session_state.get(SS_SESSION_ID, "default"),
                )
                if resp.ok and isinstance(resp.data, dict):
                    st.session_state[SS_REFINEMENT_LAB_RESULT] = resp.data
                    # append to history log
                    log: list = st.session_state.get(SS_REFINEMENT_HISTORY, [])
                    log.append(resp.data)
                    st.session_state[SS_REFINEMENT_HISTORY] = log[-20:]
                else:
                    st.error(f"Refinement service error: {resp.error}")

    result = st.session_state.get(SS_REFINEMENT_LAB_RESULT)
    if result:
        applied = result.get("applied", False)
        if not applied:
            st.info("No refinements applied — query was already optimal.")
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔤 Spelling", "📚 Synonyms", "🧠 Semantic History", "✅ Final Result"
            ])

            with tab1:
                corrections = result.get("corrections", {})
                if corrections:
                    st.markdown("**Spelling corrections applied:**")
                    rows = [{"Original": k, "Corrected": v} for k, v in corrections.items()]
                    st.table(rows)
                else:
                    st.success("No spelling errors detected.")

            with tab2:
                synonyms = result.get("synonyms_added", {})
                if synonyms:
                    st.markdown("**WordNet synonyms added:**")
                    rows = [{"Word": k, "Synonyms added": ", ".join(v)}
                            for k, v in synonyms.items()]
                    st.table(rows)
                else:
                    st.info(
                        "No synonyms added — words may be below min_term_length "
                        f"({config.REFINEMENT_MIN_TERM_LENGTH_DEFAULT} chars) or "
                        "no synsets found."
                    )

            with tab3:
                history_terms = result.get("history_terms_added", [])
                sim = result.get("history_similarity", 0.0)
                boosted = result.get("boosted_terms", [])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("History similarity", f"{sim:.3f}")
                with col_b:
                    st.metric("Terms added from history", len(history_terms))

                if sim > 0:
                    st.progress(min(sim, 1.0), text=f"Similarity: {sim:.3f}")

                if history_terms:
                    chips = "".join(f'<span class="chip">{t}</span>' for t in history_terms)
                    st.markdown("**Added terms:**")
                    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
                if boosted:
                    st.markdown(f"**Boosted terms (seen ≥2x in history):** "
                                f"{', '.join(f'`{t}`' for t in boosted)}")
                if not history_terms and not boosted:
                    st.info("No history terms added — similarity below threshold or no history provided.")

            with tab4:
                original = result.get("original_query", "")
                refined = result.get("refined_query", "")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original query**")
                    st.info(original)
                with c2:
                    st.markdown("**Refined query**")
                    st.success(refined)

                st.markdown("**Word-level diff (green = added):**")
                st.markdown(
                    f'<div class="ir-card ir-card--flat" style="font-size:1.05em;padding:12px">'
                    f'{_diff_html(original, refined)}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Refinement history log ─────────────────────────────────────────────
    history_log: list = st.session_state.get(SS_REFINEMENT_HISTORY, [])
    if history_log:
        st.markdown(f"### Session History ({len(history_log)} refinements)")
        import pandas as pd
        rows = []
        for entry in reversed(history_log):
            rows.append({
                "Original": entry.get("original_query", ""),
                "Refined": entry.get("refined_query", ""),
                "Corrections": len(entry.get("corrections", {})),
                "Synonyms": sum(len(v) for v in entry.get("synonyms_added", {}).values()),
                "History terms": len(entry.get("history_terms_added", [])),
                "Applied": "✅" if entry.get("applied") else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button("🗑️ Clear history", key="rl_clear_history"):
            st.session_state[SS_REFINEMENT_HISTORY] = []
            st.session_state[SS_REFINEMENT_LAB_RESULT] = None
            st.rerun()
