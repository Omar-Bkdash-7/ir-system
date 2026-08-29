"""Client for the Indexing microservice: build/status per index, dataset validation, dataset management."""
from __future__ import annotations

import streamlit as st

import config
from api.client import APIResponse, get, health_check, post, _request

INDEX_TYPES = ("tfidf", "bm25", "embedding")


def build_index(index_type: str, force_rebuild: bool = False, preprocessing_config: dict = None) -> APIResponse:
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type!r}")
    
    json_data = None
    if preprocessing_config:
        json_data = {"preprocessing_config": preprocessing_config}
    
    return post(
        f"{config.INDEXING_URL}/index/{index_type}/build",
        params={"force_rebuild": str(force_rebuild).lower()},
        json=json_data,
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


def index_status(index_type: str) -> APIResponse:
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type!r}")
    return get(f"{config.INDEXING_URL}/index/{index_type}/status")


def all_stats() -> APIResponse:
    return get(f"{config.INDEXING_URL}/index/stats")


def validate_dataset(dataset_id: str) -> APIResponse:
    return post(
        f"{config.INDEXING_URL}/validate",
        params={"dataset_id": dataset_id},
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


def list_local_datasets() -> APIResponse:
    return get(f"{config.INDEXING_URL}/datasets/local", max_retries=1)


def delete_local_dataset(dataset_id: str) -> APIResponse:
    return _request(
        "DELETE",
        f"{config.INDEXING_URL}/datasets/local",
        params={"dataset_id": dataset_id},
        max_retries=1,
    )


def upload_to_db(dataset_id: str) -> APIResponse:
    return post(
        f"{config.INDEXING_URL}/datasets/upload-to-db",
        params={"dataset_id": dataset_id},
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


def delete_dataset_from_db(dataset_id: str) -> APIResponse:
    """Delete dataset from PostgreSQL database."""
    return _request(
        "DELETE",
        f"{config.INDEXING_URL}/datasets/db",
        params={"dataset_id": dataset_id},
        max_retries=1,
    )


def get_db_info() -> APIResponse:
    return get(f"{config.INDEXING_URL}/datasets/db-info", max_retries=1)


def health() -> APIResponse:
    return health_check(config.INDEXING_URL)


@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def get_all_index_status_cached() -> dict:
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


# ── Inverted Index Analysis ───────────────────────────────────────────────

def get_inverted_index_stats() -> APIResponse:
    """Get comprehensive inverted index statistics."""
    return get(f"{config.INDEXING_URL}/index/inverted/stats")


def get_term_info(term: str) -> APIResponse:
    """Get detailed information about a specific term."""
    return get(f"{config.INDEXING_URL}/index/inverted/term/{term}")


def search_vocabulary(prefix: str, limit: int = 50) -> APIResponse:
    """Search vocabulary terms by prefix."""
    return get(
        f"{config.INDEXING_URL}/index/inverted/search",
        params={"prefix": prefix, "limit": limit}
    )


def get_vocabulary_sample(sample_size: int = 100) -> APIResponse:
    """Get random vocabulary sample."""
    return get(
        f"{config.INDEXING_URL}/index/inverted/sample",
        params={"sample_size": sample_size}
    )


# ── Dataset Management (Multi-Dataset Support) ────────────────────────────

def get_active_dataset() -> APIResponse:
    """Get the currently active dataset."""
    return get(f"{config.INDEXING_URL}/datasets/active", max_retries=1)


def set_active_dataset(dataset_id: str) -> APIResponse:
    """Set the active dataset."""
    return post(
        f"{config.INDEXING_URL}/datasets/active",
        params={"dataset_id": dataset_id},
        max_retries=1,
    )


def list_datasets_metadata() -> APIResponse:
    """List all datasets with their metadata."""
    return get(f"{config.INDEXING_URL}/datasets/metadata", max_retries=1)
