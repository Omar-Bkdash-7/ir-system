"""
Client for the Evaluation microservice.

The evaluation service may be a stub or briefly unavailable, so this module
also knows how to load the bundled `data/evaluation/precomputed_metrics.json`
snapshot. `pages/evaluation.py` prefers live data and falls back to this file,
showing the user which source is in play.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import config
from api.client import APIResponse, get, health_check


def health() -> APIResponse:
    return health_check(config.EVALUATION_URL)


def get_metrics() -> APIResponse:
    """GET /metrics -- live, pre-computed-per-run evaluation metrics."""
    return get(f"{config.EVALUATION_URL}/metrics")


def get_per_query_analysis() -> APIResponse:
    """GET /metrics/per-query -- best/worst performing queries per model."""
    return get(f"{config.EVALUATION_URL}/metrics/per-query")


def get_refinement_comparison() -> APIResponse:
    """GET /metrics/refinement-comparison -- metrics before vs after query refinement."""
    return get(f"{config.EVALUATION_URL}/metrics/refinement-comparison")


def load_precomputed_metrics() -> Optional[dict]:
    """Fallback: read the metrics snapshot bundled with the app, if present."""
    path = config.EVALUATION_FALLBACK_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None