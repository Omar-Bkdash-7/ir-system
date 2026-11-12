import streamlit as st
from typing import Dict, Any, Optional, List

import api.rag as rag_api
from obs import get_logger
from ui_state import dataset_selector
from utils.dataset_utils import get_available_datasets

logger = get_logger(__name__)

DOWNLOAD_POLL_SECONDS = 2

MODE_LABELS = {
    "auto": "Auto (remote, local fallback)",
    "remote": "Remote only",
    "local": "Local only",
}

REMOTE_PROBLEMS = {
    "unreachable": "cannot reach the provider (network or DNS problem)",
    "auth_error": "the token was rejected (it needs the 'Make calls to Inference Providers' permission)",
    "quota_exhausted": "the free monthly credits are exhausted",
    "model_unavailable": "the configured model is not available on the provider",
}


def render():
    st.title("RAG - Retrieval-Augmented Generation")

    st.markdown("""
    Generate natural language answers from your documents using a remote LLM
    (HuggingFace Inference Providers), with an optional local model as an offline fallback.
    """)

    # Remote/local model status comes from the RAG backend (through the API gateway)
    models = _fetch_rag_models()
    _render_models_status(models)

    st.subheader("Ask a Question")

    query = st.text_area(
        "Question",
        placeholder="e.g., What are the main findings about aerodynamics?",
        height=100,
        label_visibility="collapsed"
    )

    col_dataset, col_mode = st.columns(2)

    with col_dataset:
        available_datasets = _get_available_datasets_for_rag()
        
        if available_datasets:
            dataset = dataset_selector(
                available_datasets,
                key="rag_dataset_selector",
                label="Dataset",
                placeholder="Select a dataset...",
                help="Choose a dataset to search from",
            )
        else:
            # No datasets available
            st.selectbox(
                "Dataset",
                options=[],
                disabled=True,
                placeholder="No datasets available",
                help="Download and index datasets first"
            )
            dataset = None

    with col_mode:
        mode = st.selectbox(
            "Generation Mode",
            options=_get_available_modes(models),
            index=_get_default_mode_index(models),
            format_func=lambda m: MODE_LABELS.get(m, m),
            help="Auto tries remote first and falls back to local if it fails"
        )

    _render_local_model_panel(models)

    with st.expander("Advanced Settings", expanded=False):
        top_k = st.slider(
            "Retrieved Documents",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of documents to retrieve from the dataset"
        )

        col_a, col_b = st.columns(2)

        with col_a:
            temperature = st.slider(
                "Creativity",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Lower = more factual (recommended for RAG), higher = more creative"
            )

        with col_b:
            max_tokens = st.slider(
                "Max Length",
                min_value=100,
                max_value=800,
                value=400,
                step=50,
                help="Maximum answer length (the local model is capped at about 256 tokens)"
            )

    if st.button("Generate Answer", type="primary", use_container_width=True, disabled=not query.strip()):
        # Validate dataset selection
        if not dataset:
            st.warning("⚠️ Please select a dataset before generating an answer.")
            return
            
        with st.spinner("Retrieving documents and generating answer..."):
            response = _generate_rag_answer(
                query=query,
                dataset=dataset,
                top_k=top_k,
                mode=mode,
                temperature=temperature,
                max_tokens=max_tokens
            )

        if response:
            _display_results(response)


def _get_available_datasets_for_rag() -> List[str]:
    """Get available datasets with indices built"""
    try:
        available = get_available_datasets()
        return available if available else ["cranfield", "vaswani"]
    except Exception:  # noqa: BLE001 - the list is cosmetic, the page must still render
        logger.exception("could not list indexed datasets; falling back to defaults")
        return ["cranfield", "vaswani"]


def _fetch_rag_models() -> Optional[Dict[str, Any]]:
    response = rag_api.models()
    return response.data if response.ok and isinstance(response.data, dict) else None


def _fetch_local_model_status() -> Optional[Dict[str, Any]]:
    response = rag_api.local_model_status()
    return response.data if response.ok and isinstance(response.data, dict) else None


def _start_local_download() -> bool:
    response = rag_api.start_local_model_download()
    if response.ok:
        return True
    st.error(f"Could not start the download: {response.error}")
    return False


def _render_models_status(models: Optional[Dict[str, Any]]):
    if models is None:
        st.error(
            "RAG service is unreachable. "
            "Check that the api_gateway and rag containers are running."
        )
        return

    remote = models.get("remote_model", {})
    local = models.get("local_model", {})
    local_ready = local.get("available", False)
    
    remote_name = remote.get("name") or remote.get("id", "remote model")
    remote_state = remote.get("state", "unknown")
    
    local_name = local.get("name") or local.get("id", "local model")
    local_state = local.get("state")

    # Show models status in compact format
    col1, col2 = st.columns(2)
    
    with col1:
        if remote_state == "ok":
            st.success(f"🌐 Remote: {remote_name}")
        elif remote_state == "not_configured":
            st.error("🌐 Remote: Not configured")
        else:
            problem = REMOTE_PROBLEMS.get(remote_state, remote_state)
            st.warning(f"🌐 Remote: {problem}")
    
    with col2:
        if local_state == "ready":
            loaded = " (loaded)" if local.get("loaded") else ""
            st.success(f"💻 Local: {local_name}{loaded}")
        elif local_state == "downloading":
            st.info("💻 Local: Downloading...")
        elif local_state == "disabled":
            st.info("💻 Local: Disabled")
        elif local_state == "error":
            st.error("💻 Local: Download failed")
        else:
            col_status, col_btn = st.columns([3, 1])
            with col_status:
                st.warning("💻 Local: Not downloaded")
            with col_btn:
                if st.button("📥", key="rag_local_download_quick", help="Download local model"):
                    if _start_local_download():
                        st.rerun()

    # Show detailed warnings/errors if needed
    if remote_state == "not_configured":
        st.caption("⚠️ Set HUGGINGFACE_API_TOKEN in .env to use remote model")
    elif remote_state not in ("ok", "unknown"):
        code = f" ({remote['code']})" if remote.get("code") else ""
        if not local_ready:
            st.caption(f"⚠️ Remote unavailable{code}. Download local model (📥 button above) to work offline.")
    
    if remote.get("model_listed") is False and remote_state == "ok":
        st.caption("⚠️ Configured model not found in provider catalog")


def _get_available_modes(models: Optional[Dict[str, Any]]) -> list:
    if models is None:
        return ["auto"]
    modes = {m.get("id"): m for m in models.get("modes", [])}
    return [m for m in ("auto", "remote", "local") if modes.get(m, {}).get("available")] or ["auto"]


def _get_default_mode_index(models: Optional[Dict[str, Any]]) -> int:
    if models is None:
        return 0
    options = _get_available_modes(models)
    default_mode = models.get("default_mode", "auto")
    return options.index(default_mode) if default_mode in options else 0


def _render_local_model_panel(models: Optional[Dict[str, Any]]):
    if models is None:
        return

    local = models.get("local_model", {})
    state = local.get("state")
    name = local.get("name") or local.get("id", "local model")

    # Show download progress or error details
    if state == "downloading":
        with st.expander("📥 Download Progress", expanded=True):
            _download_progress()
    elif state == "error":
        with st.expander("❌ Download Failed", expanded=True):
            error = local.get("error") or {}
            st.error(error.get("message", "The download failed."))
            if error.get("hint"):
                st.caption(error["hint"])
            
            st.caption(
                f"{name} runs on CPU and is used automatically when the remote model is unavailable. "
                f"Download size: about {local.get('approx_size_mb', '?')} MB."
            )
            
            if st.button("🔄 Retry Download", key="rag_local_download_retry"):
                if _start_local_download():
                    st.rerun()


def _format_mb(num_bytes: Optional[int]) -> str:
    return f"{(num_bytes or 0) / (1024 * 1024):.0f} MB"


def _format_speed(bytes_per_second: int) -> str:
    if bytes_per_second >= 1024 * 1024:
        return f"{bytes_per_second / (1024 * 1024):.1f} MB/s"
    return f"{bytes_per_second / 1024:.0f} KB/s"


def _format_eta(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60} min"
    if seconds >= 60:
        return f"{seconds // 60} min"
    return f"{seconds} s"


def _render_download_progress():
    status = _fetch_local_model_status()

    if status is None:
        st.warning("Cannot read the download status right now.")
        return

    if status.get("state") != "downloading":
        # Finished (button disappears) or failed (error is shown): refresh the whole page
        st.rerun()

    phase = status.get("phase", "downloading")
    total = status.get("total_bytes")

    if phase == "verifying":
        st.progress(0.99, text="Verifying the downloaded files (SHA-256)...")
    elif phase == "finalizing":
        st.progress(0.99, text="Finalizing...")
    elif total:
        text = f"Downloading... {_format_mb(status.get('downloaded_bytes'))} / {_format_mb(total)}"
        if status.get("speed_bps"):
            text += f" - {_format_speed(status['speed_bps'])}"
        if status.get("eta_s"):
            text += f" - about {_format_eta(status['eta_s'])} left"
        st.progress(float(status.get("progress", 0.0)), text=text)
    else:
        st.progress(0.0, text="Preparing download...")

    _render_download_notices(status, phase)


def _render_download_notices(status: Dict[str, Any], phase: str):
    if phase == "retrying":
        problem = (status.get("last_error") or {}).get("message", "unknown problem")
        st.warning(
            f"Connection problem: {problem} "
            f"Retrying in {status.get('retry_in_s', 0)} s "
            f"(attempt {status.get('failures', 0)} of {status.get('max_failures', 0)} without progress). "
            "The bytes already downloaded are kept."
        )
    elif (status.get("stalled_for_s") or 0) >= 20:
        st.warning(
            f"No data received for {int(status['stalled_for_s'])} s. "
            "The download reconnects automatically and continues where it stopped."
        )
    elif status.get("slow"):
        st.caption(
            "The connection is slow. The download continues, but it may take a long time. "
            "See docs/RAG.md for faster alternatives."
        )

    if status.get("current_file") and status.get("files_total"):
        st.caption(
            f"File {min(status.get('files_done', 0) + 1, status['files_total'])} of {status['files_total']}: "
            f"{status['current_file']}"
        )


# st.fragment (Streamlit >= 1.37) or st.experimental_fragment (1.33 - 1.36): only this block re-runs every few
# seconds, not the whole page. Older versions fall back to a manual refresh button.
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

if _fragment is not None:
    _download_progress = _fragment(run_every=DOWNLOAD_POLL_SECONDS)(_render_download_progress)
else:
    def _download_progress():
        _render_download_progress()
        st.button("Refresh progress", key="rag_refresh_progress")


def _generate_rag_answer(
    query: str,
    dataset: str,
    top_k: int,
    mode: str,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    response = rag_api.generate(
        query,
        dataset,
        top_k,
        mode=mode,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if response.ok:
        return response.data

    if response.reason == "timeout":
        st.error(
            "The request timed out. The local model can be slow on CPU: "
            "try fewer documents or a shorter answer length."
        )
    elif response.status_code:
        st.error(f"Error: {response.status_code} - {response.error}")
    else:
        st.error(f"Error: {response.error}")
    return None


def _display_results(response: Dict[str, Any]):
    metadata = response.get("metadata") or {}
    status = metadata.get("status", "success")
    documents = response.get("context_documents", [])

    if status == "no_context":
        st.info(response.get("answer", "No relevant documents found."))
        return

    if status == "generation_failed":
        st.error("No model could generate an answer. The retrieved documents are shown below.")
        errors = metadata.get("errors", [])
        for attempt in errors:
            st.caption(f"{attempt.get('source', '?')}: {attempt.get('message') or attempt.get('code')}")
        if any(a.get("source") == "local" and a.get("code") == "unavailable" for a in errors):
            st.info("Tip: download the local model in the panel above to keep answering when the remote model is down.")
    else:
        st.success("Answer generated")

        model_used = response.get("model_used", "unknown")
        latency_ms = metadata.get("latency_ms")
        latency = f" - {latency_ms / 1000:.1f}s" if latency_ms else ""
        st.info(f"Model: {model_used}{latency}")

        if metadata.get("fallback_used"):
            st.warning(
                "The remote model was unavailable, so the local model answered. "
                f"Reason: {metadata.get('fallback_reason', 'unknown')}"
            )

        st.markdown("### Answer")
        st.markdown(response["answer"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", model_used.split("(")[0].strip().split("/")[-1])
        with col2:
            st.metric("Retrieval", response["retrieval_model"])
        with col3:
            st.metric("Sources", len(documents))

    if documents:
        with st.expander(f"View {len(documents)} Source Documents"):
            for idx, doc in enumerate(documents, 1):
                st.markdown(f"**{idx}. {doc['doc_id']}** (score: {doc['score']:.3f})")
                with st.container():
                    st.text(doc["text"][:400] + ("..." if len(doc["text"]) > 400 else ""))
                if idx < len(documents):
                    st.divider()


if __name__ == "__main__":
    render()
