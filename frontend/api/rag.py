"""
Client for the RAG service.

The browser never talks to the RAG container directly: every request goes
through the API gateway, so this module keeps the gateway in one place and
routes all outcomes through `api.client` (retry, timeouts, normalized errors).
"""
from __future__ import annotations

from typing import Any, Dict

import config
from api.client import APIResponse, get, health_check, post

# One generation may try the remote provider, then load and run the local model
# on CPU. Keep this slightly above the gateway's own 300 s timeout for the call.
GENERATE_TIMEOUT = 305
PROBE_TIMEOUT = 5
DOWNLOAD_START_TIMEOUT = 15


def health() -> APIResponse:
    """GET /health on the gateway."""
    return health_check(config.API_GATEWAY_URL)


def models() -> APIResponse:
    """GET /rag/models — remote provider and local model status."""
    return get(f"{config.API_GATEWAY_URL}/rag/models", timeout=PROBE_TIMEOUT)


def local_model_status() -> APIResponse:
    """GET /rag/local-model — download/availability state of the local model."""
    return get(f"{config.API_GATEWAY_URL}/rag/local-model", timeout=PROBE_TIMEOUT)


def start_local_model_download() -> APIResponse:
    """POST /rag/local-model/download — starts a background download."""
    return post(f"{config.API_GATEWAY_URL}/rag/local-model/download", timeout=DOWNLOAD_START_TIMEOUT)


def generate(
    query: str,
    dataset: str,
    top_k: int,
    *,
    mode: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 512,
    retrieval_model: str = "embedding",
    use_context: bool = True,
) -> APIResponse:
    """
    POST /rag/generate — retrieve context for `dataset` and generate an answer.

    Returns the full RAG response (`answer`, `context_documents`, `metadata`,
    `model_used`, `retrieval_model`) when `ok` is true.
    """
    payload: Dict[str, Any] = {
        "query": query,
        "dataset": dataset,
        "retrieval_model": retrieval_model,
        "top_k": top_k,
        "generation_config": {
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_context": use_context,
        },
    }
    return post(f"{config.API_GATEWAY_URL}/rag/generate", json=payload, timeout=GENERATE_TIMEOUT)
