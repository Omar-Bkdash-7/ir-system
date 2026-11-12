"""
Client for the Evaluation microservice.

The evaluation service may be a stub or briefly unavailable; `pages/evaluation.py`
prefers live data and falls back to the bundled `data/evaluation/precomputed_metrics.json`
snapshot, showing the user which source is in play.
"""
from __future__ import annotations

import config
from api.client import APIResponse, delete, get, health_check, post


def health() -> APIResponse:
    return health_check(config.EVALUATION_URL)


# A full evaluation scores every selected model over the whole query set.
EVALUATION_TIMEOUT = 6000


def run_evaluation(payload: dict) -> APIResponse:
    """POST /evaluate — run the requested models and return per-model metrics."""
    return post(f"{config.EVALUATION_URL}/evaluate", json=payload, timeout=EVALUATION_TIMEOUT)


def get_metrics() -> APIResponse:
    """GET /metrics -- live, pre-computed-per-run evaluation metrics."""
    return get(f"{config.EVALUATION_URL}/metrics")


def get_per_query_analysis() -> APIResponse:
    """GET /metrics/per-query -- best/worst performing queries per model."""
    return get(f"{config.EVALUATION_URL}/metrics/per-query")


def get_refinement_comparison() -> APIResponse:
    """GET /metrics/refinement-comparison -- metrics before vs after query refinement."""
    return get(f"{config.EVALUATION_URL}/metrics/refinement-comparison")


def get_history(limit: int = 100) -> APIResponse:
    """GET /history — stored evaluation runs, newest first."""
    return get(f"{config.EVALUATION_URL}/history", params={"limit": limit})


def delete_results(*, model: str | None = None, dataset: str | None = None) -> APIResponse:
    """DELETE /results — drop every stored run of one model or one dataset."""
    params = {name: value for name, value in (("model", model), ("dataset", dataset)) if value}
    return delete(f"{config.EVALUATION_URL}/results", params=params)


def delete_result(result_id: int | str) -> APIResponse:
    """DELETE /result/{id} — drop one stored run."""
    return delete(f"{config.EVALUATION_URL}/result/{result_id}")

