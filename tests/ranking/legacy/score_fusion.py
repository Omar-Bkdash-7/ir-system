# Snapshot of the module as it was BEFORE the rewrite, kept only as the reference for tests/ranking/test_differential.py.
# Do not import it from the service.
from typing import List
import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_list: List[List[dict]],
    k: int = 60
) -> List[dict]:
    rrf_scores = {}
    
    for model_results in results_list:
        for rank, result in enumerate(model_results, start=1):
            doc_id = result["doc_id"]
            score = 1.0 / (k + rank)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += score
    
    fused_results = [
        {"doc_id": doc_id, "score": score}
        for doc_id, score in rrf_scores.items()
    ]
    
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    logger.debug(f"RRF fusion: combined {len(results_list)} result lists, "
                f"produced {len(fused_results)} unique documents")
    
    return fused_results


def normalize_scores(results: List[dict]) -> List[dict]:
    if not results:
        return results
    
    scores = [r["score"] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        normalized = [{"doc_id": r["doc_id"], "score": 1.0} for r in results]
    else:
        normalized = [
            {
                "doc_id": r["doc_id"],
                "score": (r["score"] - min_score) / (max_score - min_score)
            }
            for r in results
        ]
    
    return normalized


def combsum_fusion(results_list: List[List[dict]]) -> List[dict]:
    normalized_lists = [normalize_scores(results) for results in results_list]
    
    combsum_scores = {}
    
    for model_results in normalized_lists:
        for result in model_results:
            doc_id = result["doc_id"]
            score = result["score"]
            
            if doc_id not in combsum_scores:
                combsum_scores[doc_id] = 0.0
            combsum_scores[doc_id] += score
    
    fused_results = [
        {"doc_id": doc_id, "score": score}
        for doc_id, score in combsum_scores.items()
    ]
    
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    logger.debug(f"CombSUM fusion: combined {len(results_list)} result lists, "
                f"produced {len(fused_results)} unique documents")
    
    return fused_results


def combmnz_fusion(results_list: List[List[dict]]) -> List[dict]:
    normalized_lists = [normalize_scores(results) for results in results_list]
    
    combmnz_scores = {}
    doc_counts = {}
    
    for model_results in normalized_lists:
        for result in model_results:
            doc_id = result["doc_id"]
            score = result["score"]
            
            if doc_id not in combmnz_scores:
                combmnz_scores[doc_id] = 0.0
                doc_counts[doc_id] = 0
            
            combmnz_scores[doc_id] += score
            doc_counts[doc_id] += 1
    
    for doc_id in combmnz_scores:
        combmnz_scores[doc_id] *= doc_counts[doc_id]
    
    fused_results = [
        {"doc_id": doc_id, "score": score}
        for doc_id, score in combmnz_scores.items()
    ]
    
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    logger.debug(f"CombMNZ fusion: combined {len(results_list)} result lists, "
                f"produced {len(fused_results)} unique documents")
    
    return fused_results


def weighted_fusion(
    results_list: List[List[dict]],
    weights: List[float]
) -> List[dict]:
    if len(results_list) != len(weights):
        raise ValueError(f"Number of result lists ({len(results_list)}) "
                        f"must match number of weights ({len(weights)})")
    
    normalized_lists = [normalize_scores(results) for results in results_list]
    
    weighted_scores = {}
    
    for weight, model_results in zip(weights, normalized_lists):
        for result in model_results:
            doc_id = result["doc_id"]
            score = result["score"] * weight
            
            if doc_id not in weighted_scores:
                weighted_scores[doc_id] = 0.0
            weighted_scores[doc_id] += score
    
    fused_results = [
        {"doc_id": doc_id, "score": score}
        for doc_id, score in weighted_scores.items()
    ]
    
    fused_results.sort(key=lambda x: x["score"], reverse=True)
    
    logger.debug(f"Weighted fusion: combined {len(results_list)} result lists "
                f"with weights {weights}, produced {len(fused_results)} unique documents")
    
    return fused_results


def fuse_results(
    results_list: List[List[dict]],
    method: str = "rrf",
    weights: List[float] = None,
    rrf_k: int = 60
) -> List[dict]:
    if not results_list:
        return []
    
    if len(results_list) == 1:
        logger.warning("Only one result list provided for fusion, returning as-is")
        return results_list[0]
    
    method = method.lower()
    
    if method == "rrf":
        return reciprocal_rank_fusion(results_list, k=rrf_k)
    elif method == "combsum":
        return combsum_fusion(results_list)
    elif method == "combmnz":
        return combmnz_fusion(results_list)
    elif method == "weighted":
        if not weights:
            raise ValueError("Weights are required for weighted fusion")
        return weighted_fusion(results_list, weights)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
