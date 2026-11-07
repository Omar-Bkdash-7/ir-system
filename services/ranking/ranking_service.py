"""Service layer of the ranking service: validates plain data, calls the rankers, times the call.

No FastAPI and no pydantic in here: everything is plain lists and dicts, so it can be tested (and reused) on its own.
The HTTP handlers in main.py only convert request models into these arguments.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from common.exceptions import InvalidQueryError
from services.ranking.rankers import DEFAULT_RRF_K, METHODS, fuse_results, stable_order

logger = logging.getLogger(__name__)

#: "cosine" is the name this endpoint always had; "score" says what it really does. Both order by the item's "score".
RANK_METHODS = ("cosine", "score")


def _scores_of(results: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The "score" of every item as float64; a missing, non-numeric or non-finite score is a bad request (400)."""
    values: List[float] = []
    for position, item in enumerate(results):
        value = item.get("score") if isinstance(item, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidQueryError(f"results[{position}] needs a numeric 'score'")
        values.append(value)
    scores = np.asarray(values, dtype=np.float64)
    if scores.size and not np.isfinite(scores).all():
        raise InvalidQueryError(f"results[{int(np.flatnonzero(~np.isfinite(scores))[0])}]: 'score' must be a finite number")
    return scores


def rank_results(results: Sequence[Mapping[str, Any]], method: str = "cosine", top_k: int = 50) -> Dict[str, Any]:
    """Order `results` by score, best first, keep `top_k`. Items are returned untouched (every extra field survives)."""
    started = time.perf_counter()
    method = str(method).strip().lower()
    if method not in RANK_METHODS:
        raise InvalidQueryError(f"Unknown ranking method: {method}")
    scores = _scores_of(results)
    order = stable_order(scores)[:top_k]  # equal scores keep the order they arrived in
    ranked = [results[position] for position in order.tolist()]
    took_ms = (time.perf_counter() - started) * 1000.0
    logger.info("rank method=%s items=%d returned=%d took_ms=%.2f", method, len(results), len(ranked), took_ms)
    return {"ranked_results": ranked, "total": len(ranked), "took_ms": round(took_ms, 3)}


def _best_first(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """RRF needs every list ordered best first: a list that is not already ordered by score is stably re-ordered."""
    if len(hits) < 2:
        return hits
    scores = np.fromiter((hit["score"] for hit in hits), dtype=np.float64, count=len(hits))
    if bool(np.all(scores[:-1] >= scores[1:])):
        return hits
    return [hits[position] for position in stable_order(scores).tolist()]


def fuse_result_lists(
    result_lists: Sequence[List[Dict[str, Any]]],
    method: str = "rrf",
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Fuse ranked lists of {"doc_id", "score"}; the best `top_k` fused documents come back, best first."""
    started = time.perf_counter()
    method = str(method).strip().lower()
    if method not in METHODS:
        raise InvalidQueryError(f"Unknown fusion method: {method}")
    if len(result_lists) < 2:
        raise InvalidQueryError("Fusion needs at least 2 result lists")
    fused = fuse_results([_best_first(hits) for hits in result_lists], method=method, weights=weights, rrf_k=rrf_k, top_k=top_k)
    took_ms = (time.perf_counter() - started) * 1000.0
    logger.info("fuse method=%s lists=%d returned=%d took_ms=%.2f", method, len(result_lists), len(fused), took_ms)
    return {"fused_results": fused, "total": len(fused), "method": method, "took_ms": round(took_ms, 3)}
