"""
Query Refinement Lab — demo and config page.

Sections, in the order they are drawn:
  1. Read-only configuration panel
  2. Live demo: inputs, analysis call, result tabs
  3. Session history table
"""
from __future__ import annotations

from typing import List

import streamlit as st

import config
from api.gateway import refine_query
from constants import (
    SS_REFINEMENT_HISTORY,
    SS_REFINEMENT_LAB_HISTORY_INPUT,
    SS_REFINEMENT_LAB_QUERY_INPUT,
    SS_REFINEMENT_LAB_RESULT,
    SS_SESSION_ID,
)
from obs import failure_hint, get_logger, log_failure
from ui_style import inject_style

logger = get_logger(__name__)


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
    return " ".join(html_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def _render_config_panel() -> None:
    """Read-only view of the server-side defaults."""
    with st.expander("⚙️ Refinement Configuration", expanded=False):
        st.caption(
            "This panel documents the active configuration. These values are informational — they reflect the server-side defaults."
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


def _render_demo_inputs() -> tuple[str, List[str]]:
    """Query + pasted history; returns both in parsed form."""
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
    history = [line.strip() for line in history_raw.splitlines() if line.strip()]
    return query_input, history


def _run_analysis(query_input: str, history: List[str]) -> None:
    """Call the refinement service and remember the result and the log entry."""
    if not query_input.strip():
        st.error("Enter a query.")
        return

    with st.spinner("Calling refinement service…"):
        resp = refine_query(
            query_input,
            history,
            enabled=True,
            session_id=st.session_state.get(SS_SESSION_ID, "default"),
        )
        if resp.ok and isinstance(resp.data, dict):
            st.session_state[SS_REFINEMENT_LAB_RESULT] = resp.data
            log: list = st.session_state.get(SS_REFINEMENT_HISTORY, [])
            log.append(resp.data)
            st.session_state[SS_REFINEMENT_HISTORY] = log[-20:]
        else:
            log_failure(logger, "refinement_lab", resp)
            st.error(f"Refinement service error: {failure_hint(resp)}")


def _render_spelling_tab(result: dict) -> None:
    corrections = result.get("corrections", {})
    if corrections:
        st.markdown("**Spelling corrections applied:**")
        rows = [{"Original": k, "Corrected": v} for k, v in corrections.items()]
        st.table(rows)
    else:
        st.success("No spelling errors detected.")


def _render_synonyms_tab(result: dict) -> None:
    synonyms = result.get("expansions", {})
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


def _render_history_tab(result: dict) -> None:
    history_terms = result.get("history_terms", [])
    sim = result.get("history_similarity", 0.0)
    boosted = list(result.get("history_boosts", {}))

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


def _render_final_tab(result: dict) -> None:
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


def _render_result() -> None:
    """Tabs explaining what each refinement stage did to the query."""
    result = st.session_state.get(SS_REFINEMENT_LAB_RESULT)
    if not result:
        return

    if not result.get("applied", False):
        st.info("No refinements applied — query was already optimal.")
        return

    inject_style("tabs")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔤 Spelling", "📚 Synonyms", "🧠 Semantic History", "✅ Final Result"
    ])
    with tab1:
        _render_spelling_tab(result)
    with tab2:
        _render_synonyms_tab(result)
    with tab3:
        _render_history_tab(result)
    with tab4:
        _render_final_tab(result)


def _render_live_demo() -> None:
    st.markdown("### Live Refinement Demo")

    query_input, history = _render_demo_inputs()
    if st.button("🔬 Analyze Refinement", type="primary", key="btn_rl_analyze"):
        _run_analysis(query_input, history)

    _render_result()


def _render_session_history() -> None:
    """Every refinement run in this session, newest first."""
    history_log: list = st.session_state.get(SS_REFINEMENT_HISTORY, [])
    if not history_log:
        return

    st.markdown(f"### Session History ({len(history_log)} refinements)")
    import pandas as pd

    rows = []
    for entry in reversed(history_log):
        rows.append({
            "Original": entry.get("original_query", ""),
            "Refined": entry.get("refined_query", ""),
            "Corrections": len(entry.get("corrections", {})),
            "Synonyms": sum(len(v) for v in entry.get("expansions", {}).values()),
            "History terms": len(entry.get("history_terms", [])),
            "Applied": "✅" if entry.get("applied") else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("🗑️ Clear history", key="rl_clear_history"):
        st.session_state[SS_REFINEMENT_HISTORY] = []
        st.session_state[SS_REFINEMENT_LAB_RESULT] = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## 🔬 Query Refinement Lab")
    st.caption(
        "Inspect and tune every stage of the query refinement pipeline: "
        "spelling correction, WordNet synonym expansion, and semantic history-based boosting."
    )

    _render_config_panel()

    st.divider()
    _render_live_demo()

    st.divider()
    _render_session_history()
