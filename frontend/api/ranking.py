"""Client for the Ranking microservice -- second-stage re-ranking used by Hybrid Serial mode."""
from __future__ import annotations

from typing import Optional

import config
from api.client import APIResponse, health_check, post


def rerank(query: str, candidate_doc_ids: list, model: str = "embedding",
           top_k: Optional[int] = None, **extra_params) -> APIResponse:
    """POST /rerank -- re-scores a candidate set with a second-stage model."""
    payload = {"query": query, "candidate_doc_ids": candidate_doc_ids, "model": model}
    if top_k is not None:
        payload["top_k"] = top_k
    payload.update(extra_params)
    return post(f"{config.RANKING_URL}/rerank", json=payload)


def health() -> APIResponse:
    return health_check(config.RANKING_URL)