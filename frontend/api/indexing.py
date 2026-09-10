"""Client for the Indexing microservice: build/status per index, dataset validation, dataset management."""
from __future__ import annotations

import streamlit as st

import config
from api.client import APIResponse, get, health_check, post, _request

INDEX_TYPES = ("tfidf", "bm25", "embedding", "all")


def build_index(index_type: str, force_rebuild: bool = False, preprocessing_config: dict = None, dataset_name: str = None) -> APIResponse:
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type!r}")
    
    # Get dataset name from gateway if not provided
    if not dataset_name:
        import api.gateway as gateway
        info = gateway.get_dataset_info_cached()
        dataset_name = info.get("dataset_name") if info else None
        if not dataset_name:
            raise ValueError("No active dataset. Please select a dataset first.")
    
    # Use build-all endpoint for 'all' type
    if index_type == "all":
        endpoint = f"{config.INDEXING_URL}/index/build-all"
    else:
        endpoint = f"{config.INDEXING_URL}/index/{index_type}/build"
    
    json_data = None
    if preprocessing_config:
        json_data = {"preprocessing_config": preprocessing_config}
    
    return post(
        endpoint,
        params={
            "dataset_name": dataset_name,
            "force_rebuild": str(force_rebuild).lower()
        },
        json=json_data,
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def index_status(index_type: str, dataset_name: str = None) -> APIResponse:
    if index_type not in INDEX_TYPES:
        raise ValueError(f"Unknown index type: {index_type!r}")
    
    # Get dataset name from gateway if not provided
    if not dataset_name:
        import api.gateway as gateway
        info = gateway.get_dataset_info_cached()
        dataset_name = info.get("dataset_name") if info else None
        if not dataset_name:
            return APIResponse(ok=False, error="No active dataset", data=None)
    
    return get(
        f"{config.INDEXING_URL}/index/{index_type}/status",
        params={"dataset_name": dataset_name}
    )


def clear_single_index_status_cache() -> None:
    index_status.clear()


def all_stats() -> APIResponse:
    return get(f"{config.INDEXING_URL}/index/stats")


def validate_dataset(dataset_id: str) -> APIResponse:
    return post(
        f"{config.INDEXING_URL}/validate",
        params={"dataset_id": dataset_id},
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )


@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def list_local_datasets() -> APIResponse:
    return get(f"{config.INDEXING_URL}/datasets/local", max_retries=1)


def clear_local_datasets_cache() -> None:
    list_local_datasets.clear()


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


@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def get_db_info() -> APIResponse:
    return get(f"{config.INDEXING_URL}/datasets/db-info", max_retries=1)


def clear_db_info_cache() -> None:
    get_db_info.clear()


def health() -> APIResponse:
    return health_check(config.INDEXING_URL)


@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def get_all_index_status_cached() -> dict:
    import api.gateway as gateway
    info = gateway.get_dataset_info_cached()
    dataset_name = info.get("dataset_name") if info else None
    
    summary = {}
    for index_type in INDEX_TYPES:
        if index_type == "all":
            continue
        resp = index_status(index_type, dataset_name)
        if resp.ok and isinstance(resp.data, dict):
            summary[index_type] = resp.data
        else:
            summary[index_type] = {"status": "unknown", "error": resp.error}
    return summary


def clear_index_status_cache() -> None:
    """Clear all index-related caches."""
    get_all_index_status_cached.clear()
    index_status.clear()
    list_local_datasets.clear()
    get_db_info.clear()
    list_datasets_metadata.clear()


# ── Inverted Index Analysis ───────────────────────────────────────────────

@st.cache_data(ttl=config.CACHE_TTL_INDEX_STATUS, show_spinner=False)
def get_inverted_index_stats() -> APIResponse:
    """Get comprehensive inverted index statistics (cached)."""
    return get(f"{config.INDEXING_URL}/index/inverted/stats")


def clear_inverted_index_stats_cache() -> None:
    get_inverted_index_stats.clear()


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

@st.cache_data(ttl=config.CACHE_TTL_DATASET_INFO, show_spinner=False)
def list_datasets_metadata() -> APIResponse:
    """List all datasets with their metadata."""
    return get(f"{config.INDEXING_URL}/datasets/metadata", max_retries=1)


def clear_datasets_metadata_cache() -> None:
    list_datasets_metadata.clear()


def fix_corrupted_dataset(dataset_id: str) -> APIResponse:
    """Fix corrupted dataset by deleting and re-downloading."""
    return post(
        f"{config.INDEXING_URL}/datasets/fix-corrupted",
        params={"dataset_id": dataset_id},
        timeout=config.BUILD_TIMEOUT,
        max_retries=1,
    )

