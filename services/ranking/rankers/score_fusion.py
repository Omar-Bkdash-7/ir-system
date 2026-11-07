"""Rank fusion: combine several ranked result lists into one list (RRF, CombSUM, CombMNZ, weighted).

The four methods share one vectorised engine. The public functions keep the signatures and the exact output of the
pure-Python code they replaced: the same documents, bit-identical scores, and the same order for equal scores (a
document's first appearance across the lists, in list order).

Input: lists of {"doc_id": ..., "score": ...} mappings, each list ordered best first (RRF uses the position in the list
as the rank). Output: new {"doc_id", "score"} dicts, best first. A document that occurs twice in one list counts twice.
Nothing here reads the database or keeps state, so it is safe to call from any thread (retrieval calls it in-process).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

METHODS = ("rrf", "combsum", "combmnz", "weighted")
DEFAULT_RRF_K = 60

Hit = Mapping[str, Any]
ResultLists = Sequence[Sequence[Hit]]


class _Pool(NamedTuple):
    doc_ids: List[Any]        # distinct documents, in order of first appearance across the lists
    rows: List[np.ndarray]    # per list: index into doc_ids of each of its hits
    scores: List[np.ndarray]  # per list: its scores as float64, in list order


def _pool(results_list: ResultLists, with_scores: bool = True) -> _Pool:
    """Give every distinct document a row number (first appearance = lowest) and turn each list into arrays.

    `with_scores=False` skips the score column entirely: RRF only needs positions, so it never reads `score`."""
    index: Dict[Any, int] = {}
    rows: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for hits in results_list:
        doc_ids = [hit["doc_id"] for hit in hits]
        rows.append(np.array([index.setdefault(doc_id, len(index)) for doc_id in doc_ids], dtype=np.int64))
        if with_scores:
            scores.append(np.array([hit["score"] for hit in hits], dtype=np.float64))
    return _Pool(list(index), rows, scores)


def _sum_by_doc(pool: _Pool, per_list_values: Sequence[np.ndarray]) -> np.ndarray:
    """Sum every document's values over the lists, adding in list order then position (the order the old loops used)."""
    if not pool.doc_ids:
        return np.zeros(0, dtype=np.float64)
    return np.bincount(np.concatenate(pool.rows), weights=np.concatenate(per_list_values), minlength=len(pool.doc_ids))


def _minmax(values: np.ndarray) -> np.ndarray:
    """(v - min) / (max - min); a list whose scores are all equal becomes all 1.0."""
    if values.size == 0:
        return values
    low, high = values.min(), values.max()
    if high == low:
        return np.ones_like(values)
    return (values - low) / (high - low)


def _ranked(pool: _Pool, fused: np.ndarray, top_k: Optional[int]) -> List[Dict[str, Any]]:
    order = np.argsort(-fused, kind="stable")  # stable: equal scores stay in first-appearance order
    if top_k is not None:
        order = order[:top_k]
    doc_ids = pool.doc_ids
    return [{"doc_id": doc_ids[row], "score": score} for row, score in zip(order.tolist(), fused[order].tolist())]


def _check_rrf_k(k: float) -> None:
    if k < 0:
        raise ValueError(f"rrf_k must be >= 0, got {k}")


# --------------------------------------------------------------------------------------------------------------------
# the four methods
# --------------------------------------------------------------------------------------------------------------------
def reciprocal_rank_fusion(results_list: ResultLists, k: int = DEFAULT_RRF_K, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """score(d) = sum over the lists of 1 / (k + rank of d), rank starting at 1. Ignores the scores themselves."""
    _check_rrf_k(k)
    pool = _pool(results_list, with_scores=False)
    contributions = [1.0 / (k + np.arange(1, rows.size + 1)) for rows in pool.rows]
    return _ranked(pool, _sum_by_doc(pool, contributions), top_k)


def combsum_fusion(results_list: ResultLists, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Min-max normalise every list, then add a document's normalised scores."""
    pool = _pool(results_list)
    return _ranked(pool, _sum_by_doc(pool, [_minmax(scores) for scores in pool.scores]), top_k)


def combmnz_fusion(results_list: ResultLists, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """CombSUM multiplied by the number of lists the document appears in."""
    pool = _pool(results_list)
    if not pool.doc_ids:
        return []
    sums = _sum_by_doc(pool, [_minmax(scores) for scores in pool.scores])
    counts = np.bincount(np.concatenate(pool.rows), minlength=len(pool.doc_ids))
    return _ranked(pool, sums * counts, top_k)


def weighted_fusion(results_list: ResultLists, weights: Sequence[float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Min-max normalise every list, multiply list i by weights[i], then add."""
    if len(results_list) != len(weights):
        raise ValueError(f"Number of result lists ({len(results_list)}) must match number of weights ({len(weights)})")
    pool = _pool(results_list)
    return _ranked(pool, _sum_by_doc(pool, [_minmax(scores) * weight for scores, weight in zip(pool.scores, weights)]), top_k)


def normalize_scores(results: Sequence[Hit]) -> List[Dict[str, Any]]:
    """Min-max normalisation of ONE list into new {"doc_id", "score"} dicts; all-equal scores become 1.0."""
    if not results:
        return results  # type: ignore[return-value]  # an empty list comes back as it is
    scores = np.fromiter((hit["score"] for hit in results), dtype=np.float64, count=len(results))
    return [{"doc_id": hit["doc_id"], "score": value} for hit, value in zip(results, _minmax(scores).tolist())]


def fuse_results(
    results_list: ResultLists,
    method: str = "rrf",
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fuse `results_list` with `method` ("rrf", "combsum", "combmnz" or "weighted").

    No lists -> []. One list -> that very list, unchanged (nothing to fuse). `weights` is only used by "weighted" and
    must then have one entry per list. `top_k` (new, optional) keeps only the best top_k fused documents.
    Raises ValueError for an unknown method, missing or mismatched weights, or a negative rrf_k.
    """
    if not results_list:
        return []
    if len(results_list) == 1:
        logger.warning("Only one result list provided for fusion, returning as-is")
        only = results_list[0]
        return only if top_k is None else list(only[:top_k])  # type: ignore[return-value]
    method = method.lower()
    if method == "rrf":
        return reciprocal_rank_fusion(results_list, k=rrf_k, top_k=top_k)
    if method == "combsum":
        return combsum_fusion(results_list, top_k=top_k)
    if method == "combmnz":
        return combmnz_fusion(results_list, top_k=top_k)
    if method == "weighted":
        if not weights:
            raise ValueError("Weights are required for weighted fusion")
        return weighted_fusion(results_list, weights, top_k=top_k)
    raise ValueError(f"Unknown fusion method: {method}")
