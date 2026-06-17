import logging

logger = logging.getLogger(__name__)

def rank_by_cosine(results: list[dict], top_k: int = None) -> list[dict]:
    if not results:
        return []
    
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    if top_k is not None:
        sorted_results = sorted_results[:top_k]
    
    logger.info(f"Ranked {len(sorted_results)} documents by cosine similarity")
    
    return sorted_results
