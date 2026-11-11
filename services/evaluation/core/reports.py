"""Shapes the HTTP layer returns that are derived from stored rows."""
from __future__ import annotations

from typing import Any, Dict, List

from .names import display_label


def latest_metrics_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Latest stored run of every model, keyed by its display label.

    ``rows`` are expected newest-first (as `ResultsStorage.get_latest_results`
    returns them), so the first row seen for a model is its most recent run.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        model = row.get("model") or ""
        key = display_label(model)
        if key in latest:
            continue
        latest[key] = {
            "map": row.get("map", 0.0),
            "ndcg_at_10": row.get("ndcg", 0.0),
            "recall": row.get("recall", 0.0),
            "precision_at_10": row.get("precision", 0.0),
            "num_queries": row.get("num_queries", 0),
        }
    return latest


def per_query_analysis() -> Dict[str, Any]:
    """
    Best/worst queries per model — not implemented.

    The stored rows keep only aggregate metrics (`include_per_query` results are
    not persisted), so there is nothing to analyse yet. The endpoint reports that
    honestly instead of returning an empty list that looks like "no queries".
    """
    return {
        "message": "Per-query analysis not yet implemented",
        "available": False,
    }


def refinement_comparison() -> Dict[str, Any]:
    """
    Metrics before/after query refinement — not implemented server-side.

    The frontend derives this from `/history` by pairing runs of the same model
    that differ in `metadata.enable_refinement`, so the endpoint says so rather
    than duplicating that logic with a second, divergent implementation.
    """
    return {
        "message": "Refinement comparison requires separate evaluation runs",
        "available": False,
        "before": {},
        "after": {},
    }
