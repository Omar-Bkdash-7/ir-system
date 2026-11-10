"""
Client for the Clustering service.

Endpoints (services/clustering/main.py):
  GET    /status/{dataset}         cheap summary (no label arrays)
  GET    /clusters/{dataset}       full result, optionally with the 2D projection
  POST   /cluster                  run (or reuse a cached run) for one dataset
  DELETE /clusters/{dataset}       drop the stored run
  GET    /datasets                 datasets that currently have a stored run
  GET    /health                   liveness + `clustering_available`
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import config
from api.client import APIResponse, delete, get, health_check, post

CLUSTER_TIMEOUT = 60          # cached run / HDBSCAN on a few thousand documents
CLUSTER_LONG_TIMEOUT = 600    # cold run with the 2D projection
STATUS_TIMEOUT = 10
RESULT_TIMEOUT = 30


def health() -> APIResponse:
    return health_check(config.CLUSTERING_URL)


def status(dataset: str) -> APIResponse:
    """GET /status/{dataset} — small JSON, safe to poll."""
    return get(f"{config.CLUSTERING_URL}/status/{dataset}", timeout=STATUS_TIMEOUT)


def clusters(dataset: str, *, include_visualization: bool = True) -> APIResponse:
    """GET /clusters/{dataset} — stored labels, topics and (optionally) the projection."""
    return get(
        f"{config.CLUSTERING_URL}/clusters/{dataset}",
        params={"include_visualization": str(include_visualization).lower()},
        timeout=RESULT_TIMEOUT,
    )


def document(dataset: str, doc_id: str) -> APIResponse:
    """GET /document/{dataset}/{doc_id} — the cluster a single document belongs to."""
    return get(f"{config.CLUSTERING_URL}/document/{dataset}/{doc_id}")


def datasets() -> APIResponse:
    """GET /datasets — datasets with a stored clustering run."""
    return get(f"{config.CLUSTERING_URL}/datasets", timeout=STATUS_TIMEOUT)


def run(
    dataset: str,
    *,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    metric: Optional[str] = None,
    n_keywords: Optional[int] = None,
    force_recompute: bool = False,
    include_visualization: bool = False,
    long_run: bool = False,
) -> APIResponse:
    """POST /cluster — computes (or reuses) the clustering of one dataset."""
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "force_recompute": force_recompute,
        "include_visualization": include_visualization,
    }
    if min_cluster_size is not None:
        payload["min_cluster_size"] = min_cluster_size
    if min_samples is not None:
        payload["min_samples"] = min_samples
    if metric is not None:
        payload["metric"] = metric
    if n_keywords is not None:
        payload["n_keywords"] = n_keywords

    return post(
        f"{config.CLUSTERING_URL}/cluster",
        json=payload,
        timeout=CLUSTER_LONG_TIMEOUT if long_run else CLUSTER_TIMEOUT,
    )


def delete_run(dataset: str) -> APIResponse:
    """DELETE /clusters/{dataset}."""
    return delete(f"{config.CLUSTERING_URL}/clusters/{dataset}")
