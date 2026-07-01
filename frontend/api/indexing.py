"""Client for the Indexing microservice: build/status per index, dataset validation."""
from __future__ import annotations

import streamlit as st

import config
from api.client import APIResponse, get, health_check, post

INDEX_TYPES = ("tfidf", "bm25", "embedding")


def build_index(index_type: str, force_rebuild: bool = False) -> APIResponse:
    """POST /index/{type}/build?force_rebuild={bool}. Builds can take minutes."""
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type!r}")
    return post(
        f"{config.INDEXING_URL}/index/{index_type}/build",
        params={"force_rebuild": str(force_rebuild).lower()},
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


def index_status(index_type: str) -> APIResponse:
    """GET /index/{type}/status"""
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type!r}")
    return get(f"{config.INDEXING_URL}/index/{index_type}/status")


def all_stats() -> APIResponse:
    """GET /index/stats -- all three indices in one call."""
    return get(f"{config.INDEXING_URL}/index/stats")


def validate_dataset(dataset_id: str) -> APIResponse:
    """POST /validate?dataset_id=... -- can be slow; caller should poll or show a spinner."""
    return post(
        f"{config.INDEXING_URL}/validate",
        params={"dataset_id": dataset_id},
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


def health() -> APIResponse:
    return health_check(config.INDEXING_URL)


@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def get_all_index_status_cached() -> dict:
    """
    Cached summary used by the sidebar (always visible) so it doesn't hammer
    the indexing service on every rerun. The Index Management page uses the
    uncached functions above directly so it always shows fresh data.
    """
    summary = {}
    for index_type in INDEX_TYPES:
        resp = index_status(index_type)
        if resp.ok and isinstance(resp.data, dict):
            summary[index_type] = resp.data
        else:
            summary[index_type] = {"status": "unknown", "error": resp.error}
    return summary


def clear_index_status_cache() -> None:
    get_all_index_status_cached.clear()