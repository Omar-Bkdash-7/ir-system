"""Client for the Retrieval microservice -- direct status checks per retriever."""
from __future__ import annotations

import config
from api.client import APIResponse, get, health_check


def status() -> APIResponse:
    """GET /status -- returns status for all three retrievers (TF-IDF, BM25, Embedding) in one call."""
    return get(f"{config.RETRIEVAL_URL}/status")


def health() -> APIResponse:
    return health_check(config.RETRIEVAL_URL)