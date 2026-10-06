import streamlit as st
import httpx
import os
from typing import Dict, Any

API_URL = os.getenv("API_GATEWAY_URL", "http://api_gateway:8000")


def render_rag_page():
    st.title("RAG - Retrieval-Augmented Generation")

    st.markdown("""
    Generate natural language answers from your documents using the HuggingFace Inference API.
    """)

    # Check API token status from backend health endpoint
    hf_status = _check_hf_token_status()

    if hf_status.get("configured"):
        st.success("HuggingFace API: configured")
    else:
        st.error(
            "HuggingFace API token not configured. "
            "Set HUGGINGFACE_API_TOKEN in your .env file to use RAG."
        )

    st.subheader("Ask a Question")

    col_query, col_settings = st.columns([2, 1])

    with col_query:
        query = st.text_area(
            "Question",
            placeholder="e.g., What are the main findings about aerodynamics?",
            height=100,
            label_visibility="collapsed"
        )

        dataset = st.selectbox(
            "Dataset",
            options=["cranfield", "vaswani"],
            help="Select dataset to search"
        )

    with col_settings:
        st.markdown("**Retrieval Settings**")

        retrieval_model = st.selectbox(
            "Method",
            options=["embedding", "tfidf", "bm25"],
            index=0,
            help="Retrieval algorithm"
        )

        top_k = st.slider(
            "Documents",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of documents to retrieve"
        )

    with st.expander("Advanced Settings", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            temperature = st.slider(
                "Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative"
            )

        with col_b:
            max_tokens = st.slider(
                "Max Length",
                min_value=100,
                max_value=800,
                value=400,
                step=50,
                help="Maximum answer length"
            )

        if retrieval_model == "bm25":
            st.markdown("**BM25 Parameters**")
            col_bm1, col_bm2 = st.columns(2)
            with col_bm1:
                bm25_k1 = st.slider("k1", 0.0, 3.0, 1.5, 0.1)
            with col_bm2:
                bm25_b = st.slider("b", 0.0, 1.0, 0.75, 0.05)
        else:
            bm25_k1, bm25_b = 1.5, 0.75

    if st.button("Generate Answer", type="primary", use_container_width=True, disabled=not query.strip()):
        with st.spinner("Retrieving documents and generating answer..."):
            response = _generate_rag_answer(
                query=query,
                dataset=dataset,
                retrieval_model=retrieval_model,
                top_k=top_k,
                llm_model="flan-t5-large",
                temperature=temperature,
                max_tokens=max_tokens,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b
            )

        if response:
            _display_results(response)


def _check_hf_token_status() -> Dict[str, Any]:
    try:
        response = httpx.get(f"{API_URL}/rag/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            deps = data.get("dependencies", {})
            return {
                "configured": deps.get("huggingface_token") == "configured",
                "remote_available": deps.get("remote_available", False)
            }
    except Exception:
        pass
    return {"configured": False, "remote_available": False}


def _generate_rag_answer(
    query: str,
    dataset: str,
    retrieval_model: str,
    top_k: int,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75
) -> Dict[str, Any]:
    try:
        payload = {
            "query": query,
            "dataset": dataset,
            "retrieval_model": retrieval_model,
            "top_k": top_k,
            "generation_config": {
                "model": llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "use_context": True
            },
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b
        }

        response = httpx.post(
            f"{API_URL}/rag/generate",
            json=payload,
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        st.error(f"Error: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def _display_results(response: Dict[str, Any]):
    st.success("Answer generated")

    model_used = response.get("model_used", "unknown")
    st.info(f"Model: {model_used}")

    st.markdown("### Answer")
    st.markdown(response["answer"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_used.split("(")[0].strip())
    with col2:
        st.metric("Retrieval", response["retrieval_model"])
    with col3:
        st.metric("Sources", len(response["context_documents"]))

    if response["context_documents"]:
        with st.expander(f"View {len(response['context_documents'])} Source Documents"):
            for idx, doc in enumerate(response["context_documents"], 1):
                st.markdown(f"**{idx}. {doc['doc_id']}** (score: {doc['score']:.3f})")
                with st.container():
                    st.text(doc["text"][:400] + ("..." if len(doc["text"]) > 400 else ""))
                if idx < len(response["context_documents"]):
                    st.divider()


def render():
    render_rag_page()


if __name__ == "__main__":
    render_rag_page()
