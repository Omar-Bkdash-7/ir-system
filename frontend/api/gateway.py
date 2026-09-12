"""
Client for the API Gateway — single entry point for all searches and query refinement.

Endpoints (exact match with backend services/api_gateway/main.py):
  GET  /health
  GET  /datasets/info
  POST /query/refine
  POST /search                    ← individual model search
  POST /search/hybrid/serial
  POST /search/hybrid/parallel
"""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

import config
from api.client import APIResponse, get, health_check, post


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health() -> APIResponse:
    return health_check(config.API_GATEWAY_URL)


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------

def get_dataset_info() -> APIResponse:
    """GET /datasets/info — reads dataset metadata from the backend."""
    return get(f"{config.API_GATEWAY_URL}/datasets/info")


@st.cache_data(ttl=config.CACHE_TTL_DATASET_INFO, show_spinner=False)
def get_dataset_info_cached() -> Optional[dict]:
    """Cached summary used by the always-visible sidebar."""
    resp = get_dataset_info()
    return resp.data if resp.ok and isinstance(resp.data, dict) else None


def clear_dataset_info_cache() -> None:
    get_dataset_info_cached.clear()


# ---------------------------------------------------------------------------
# Query refinement  POST /query/refine
# ---------------------------------------------------------------------------

def refine_query(
    query: str,
    history: List[str],
    enabled: bool = True,
    session_id: str = "default",
) -> APIResponse:
    """
    POST /query/refine

    Response fields: original_query, refined_query, corrections{},
    synonyms_added{}, history_terms_added[], history_similarity,
    boosted_terms[], applied
    """
    return post(
        f"{config.API_GATEWAY_URL}/query/refine",
        json={
            "query": query,
            "history": history,
            "enabled": enabled,
            "session_id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# Individual model search  POST /search
# ---------------------------------------------------------------------------

def search_basic(
    query: str,
    model: str,
    top_k: int = 50,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    enable_refinement: bool = False,
    query_history: Optional[List[str]] = None,
    dataset: Optional[str] = None,
) -> APIResponse:
    """
    POST /search
    model: "tfidf" | "embedding" | "bm25"
    dataset: dataset name (e.g., 'cranfield', 'vaswani')
    """
    payload = {
        "query": query,
        "model": model,
        "top_k": top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "enable_refinement": enable_refinement,
        "query_history": query_history or [],
    }
    
    if dataset:
        payload["dataset"] = dataset
    
    return post(
        f"{config.API_GATEWAY_URL}/search",
        json=payload,
    )


# ---------------------------------------------------------------------------
# Hybrid Serial  POST /search/hybrid/serial
# ---------------------------------------------------------------------------

def search_hybrid_serial(
    query: str,
    first_model: str,
    second_model: str,
    first_stage_k: int = 1000,
    final_top_k: int = 50,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    enable_refinement: bool = False,
    query_history: Optional[List[str]] = None,
    dataset: Optional[str] = None,
) -> APIResponse:
    """POST /search/hybrid/serial"""
    payload = {
        "query": query,
        "first_model": first_model,
        "second_model": second_model,
        "first_stage_k": first_stage_k,
        "final_top_k": final_top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "enable_refinement": enable_refinement,
        "query_history": query_history or [],
    }
    
    if dataset:
        payload["dataset"] = dataset
    
    return post(
        f"{config.API_GATEWAY_URL}/search/hybrid/serial",
        json=payload,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Hybrid Parallel  POST /search/hybrid/parallel
# ---------------------------------------------------------------------------

def search_hybrid_parallel(
    query: str,
    models: List[str],
    fusion_method: str = "rrf",
    top_k: int = 50,
    candidates_per_model: int = 1000,
    rrf_k: int = 60,
    model_weights: Optional[Dict[str, float]] = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    enable_refinement: bool = False,
    query_history: Optional[List[str]] = None,
    dataset: Optional[str] = None,
) -> APIResponse:
    """POST /search/hybrid/parallel"""
    payload: dict = {
        "query": query,
        "models": models,
        "fusion_method": fusion_method,
        "top_k": top_k,
        "candidates_per_model": candidates_per_model,
        "rrf_k": rrf_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "enable_refinement": enable_refinement,
        "query_history": query_history or [],
    }
    if fusion_method == "weighted" and model_weights:
        payload["model_weights"] = model_weights
    
    if dataset:
        payload["dataset"] = dataset
    
    return post(
        f"{config.API_GATEWAY_URL}/search/hybrid/parallel",
        json=payload,
        timeout=90,
    )
