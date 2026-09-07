"""
Inverted Index Explorer page - detailed analysis of BM25 inverted index structure.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

import api.indexing as indexing_api


def _render_index_overview(cached_stats: dict | None = None) -> None:
    """Render overall inverted index statistics."""
    st.markdown("### 📊 Index Overview")
    
    # Use provided cached stats or fetch if not provided
    if cached_stats is None:
        resp = indexing_api.get_inverted_index_stats()
        if not resp.ok:
            st.error(f"Failed to load index stats: {resp.error}")
            return
        data = resp.data
    else:
        data = cached_stats
    
    if not isinstance(data, dict):
        st.error("Invalid response format")
        return
    
    if data.get("status") == "not_built":
        st.warning("⚠️ BM25 index has not been built yet. Go to Index Management to build it.")
        return
    
    if data.get("status") == "error":
        st.error(f"Error: {data.get('message', 'Unknown error')}")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vocabulary Size", f"{data.get('vocab_size', 0):,}")
    
    with col2:
        st.metric("Total Documents", f"{data.get('total_docs', 0):,}")
    
    with col3:
        st.metric("Avg Doc Length", f"{data.get('avg_doc_length', 0):.1f}")
    
    with col4:
        st.metric("Total Postings", f"{data.get('total_postings', 0):,}")
    
    st.metric("Avg Postings per Term", f"{data.get('avg_postings_per_term', 0):.2f}")
    
    if "doc_length_stats" in data and data["doc_length_stats"]:
        st.markdown("#### Document Length Distribution")
        doc_stats = data["doc_length_stats"]
        
        cols = st.columns(5)
        cols[0].metric("Min", doc_stats.get("min", 0))
        cols[1].metric("25th %ile", doc_stats.get("percentile_25", 0))
        cols[2].metric("Median", doc_stats.get("median", 0))
        cols[3].metric("75th %ile", doc_stats.get("percentile_75", 0))
        cols[4].metric("Max", doc_stats.get("max", 0))


def _render_top_terms(cached_stats: dict | None = None) -> None:
    """Render top terms by document frequency."""
    st.markdown("### 🔝 Top Terms by Document Frequency")
    
    # Use provided cached stats or fetch if not provided
    if cached_stats is None:
        resp = indexing_api.get_inverted_index_stats()
        if not resp.ok or not isinstance(resp.data, dict):
            return
        data = resp.data
    else:
        data = cached_stats
    
    if data.get("status") != "success":
        return
    
    top_terms = data.get("top_terms_by_df", [])
    
    if not top_terms:
        st.info("No terms available")
        return
    
    df = pd.DataFrame(top_terms)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "term": st.column_config.TextColumn("Term", width="medium"),
            "doc_freq": st.column_config.NumberColumn("Doc Frequency", format="%d"),
            "idf": st.column_config.NumberColumn("IDF Score", format="%.4f")
        },
        hide_index=True
    )


def _render_longest_posting_lists(cached_stats: dict | None = None) -> None:
    """Render terms with longest posting lists."""
    st.markdown("### 📋 Longest Posting Lists")
    
    # Use provided cached stats or fetch if not provided
    if cached_stats is None:
        resp = indexing_api.get_inverted_index_stats()
        if not resp.ok or not isinstance(resp.data, dict):
            return
        data = resp.data
    else:
        data = cached_stats
    
    if data.get("status") != "success":
        return
    
    longest = data.get("longest_posting_lists", [])
    
    if not longest:
        st.info("No data available")
        return
    
    df = pd.DataFrame(longest)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "term": st.column_config.TextColumn("Term", width="medium"),
            "posting_count": st.column_config.NumberColumn("Posting Count", format="%d")
        },
        hide_index=True
    )


def _render_term_search() -> None:
    """Render term search interface."""
    st.markdown("### 🔍 Term Search")
    
    tab1, tab2, tab3 = st.tabs(["Search by Prefix", "Lookup Specific Term", "Random Sample"])
    
    with tab1:
        st.markdown("Search for terms starting with a specific prefix")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            prefix = st.text_input(
                "Prefix",
                value="",
                placeholder="e.g., 'data', 'comp', 'query'",
                key="prefix_search"
            )
        
        with col2:
            limit = st.number_input(
                "Limit",
                min_value=10,
                max_value=200,
                value=50,
                key="prefix_limit"
            )
        
        if prefix and len(prefix) > 0:
            resp = indexing_api.search_vocabulary(prefix, limit)
            
            if resp.ok and isinstance(resp.data, dict):
                data = resp.data
                
                if data.get("status") == "success":
                    results = data.get("results", [])
                    total_matches = data.get("total_matches", 0)
                    
                    st.info(f"Found {total_matches} terms starting with '{prefix}' (showing {len(results)})")
                    
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "term": st.column_config.TextColumn("Term"),
                                "doc_freq": st.column_config.NumberColumn("Doc Freq", format="%d"),
                                "idf": st.column_config.NumberColumn("IDF", format="%.4f"),
                                "posting_count": st.column_config.NumberColumn("Postings", format="%d")
                            },
                            hide_index=True
                        )
                else:
                    st.warning(data.get("message", "No results"))
            else:
                st.error(f"Search failed: {resp.error}")
    
    with tab2:
        st.markdown("Get detailed information about a specific term")
        
        term = st.text_input(
            "Term to lookup",
            value="",
            placeholder="e.g., 'database', 'information', 'retrieval'",
            key="term_lookup"
        )
        
        if term and len(term) > 0:
            resp = indexing_api.get_term_info(term.lower())
            
            if resp.ok and isinstance(resp.data, dict):
                data = resp.data
                
                if data.get("status") == "success":
                    if data.get("exists"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Document Frequency", f"{data.get('doc_frequency', 0):,}")
                        
                        with col2:
                            st.metric("IDF Score", f"{data.get('idf', 0):.4f}")
                        
                        with col3:
                            st.metric("Total Postings", f"{data.get('total_postings', 0):,}")
                        
                        posting_sample = data.get("posting_list_sample", [])
                        
                        if posting_sample:
                            st.markdown("#### Posting List Sample")
                            st.caption(f"Showing first {data.get('sample_size', 0)} of {data.get('total_size', 0)} postings")
                            
                            df = pd.DataFrame(posting_sample)
                            st.dataframe(
                                df,
                                use_container_width=True,
                                column_config={
                                    "doc_id": st.column_config.TextColumn("Document ID"),
                                    "term_freq": st.column_config.NumberColumn("Term Frequency", format="%d")
                                },
                                hide_index=True,
                                height=400
                            )
                    else:
                        st.warning(f"Term '{term}' not found in vocabulary")
                else:
                    st.error(data.get("message", "Unknown error"))
            else:
                st.error(f"Lookup failed: {resp.error}")
    
    with tab3:
        st.markdown("Get a random sample of vocabulary terms")
        
        sample_size = st.slider(
            "Sample Size",
            min_value=10,
            max_value=500,
            value=100,
            key="sample_size"
        )
        
        if st.button("🎲 Get Random Sample", key="get_sample"):
            resp = indexing_api.get_vocabulary_sample(sample_size)
            
            if resp.ok and isinstance(resp.data, dict):
                data = resp.data
                
                if data.get("status") == "success":
                    vocab_size = data.get("vocab_size", 0)
                    sample = data.get("sample", [])
                    
                    st.info(f"Sampled {len(sample)} terms from vocabulary of {vocab_size:,}")
                    
                    if sample:
                        df = pd.DataFrame(sample)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "term": st.column_config.TextColumn("Term"),
                                "doc_freq": st.column_config.NumberColumn("Doc Freq", format="%d"),
                                "idf": st.column_config.NumberColumn("IDF", format="%.4f")
                            },
                            hide_index=True,
                            height=400
                        )
                else:
                    st.warning(data.get("message", "Error loading sample"))
            else:
                st.error(f"Failed to get sample: {resp.error}")


def render() -> None:
    """Main render function for Inverted Index Explorer page."""
    st.markdown("## 🗃️ Inverted Index Explorer")
    st.caption("Deep dive into BM25 inverted index structure and vocabulary")
    
    st.divider()
    
    # Fetch stats once and reuse for all sections
    resp = indexing_api.get_inverted_index_stats()
    cached_stats = resp.data if resp.ok and isinstance(resp.data, dict) else None
    
    _render_index_overview(cached_stats)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        _render_top_terms(cached_stats)
    
    with col2:
        _render_longest_posting_lists(cached_stats)
    
    st.divider()
    
    _render_term_search()
