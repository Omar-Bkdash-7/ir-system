import httpx
from common.config import get_config
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

PREPROCESSING_SERVICE_URL = get_config("services.preprocessing_url", "http://localhost:8001")
RETRIEVAL_SERVICE_URL = get_config("services.retrieval_url", "http://localhost:8003")


def _get_preprocessing_config_for_dataset(dataset: Optional[str]) -> Optional[dict]:
    """
    Retrieve saved preprocessing config for a dataset.
    Returns None if dataset not specified or config not found.
    """
    if not dataset:
        return None
    
    try:
        from services.indexing.dataset.service import DatasetService
        dataset_service = DatasetService()
        config = dataset_service.get_preprocessing_config(dataset)
        if config:
            logger.info(f"Using saved preprocessing config for dataset: {dataset}")
        return config
    except Exception as e:
        logger.warning(f"Could not load preprocessing config for dataset {dataset}: {e}")
        return None


def _preprocess_query(query: str, dataset: Optional[str] = None) -> List[str]:
    """
    Preprocess query with dataset-specific config if available.
    
    Args:
        query: Query text
        dataset: Dataset name (optional)
    
    Returns:
        List of query tokens
    """
    preprocessing_config = _get_preprocessing_config_for_dataset(dataset)
    
    preprocess_payload = {
        "text": query,
        "mode": "query"
    }
    
    if preprocessing_config:
        preprocess_payload["config"] = preprocessing_config
    
    with httpx.Client(timeout=30.0) as client:
        preprocess_resp = client.post(
            f"{PREPROCESSING_SERVICE_URL}/preprocess",
            json=preprocess_payload
        )
        preprocess_resp.raise_for_status()
        preprocess_data = preprocess_resp.json()
        
        return preprocess_data.get("tokens", [])


def search_basic(
    query: str,
    model: str = "tfidf",
    top_k: int = 50,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    dataset: str = None
) -> dict:
    logger.info(f"Basic search: model={model}, top_k={top_k}, dataset={dataset}")
    logger.info(f"Query: '{query[:50]}...'")
    
    try:
        # Preprocess query with dataset-specific config
        query_tokens = _preprocess_query(query, dataset)
        
        if not query_tokens:
            logger.warning("Preprocessing returned empty tokens")
            return {
                "results": [],
                "total": 0,
                "model": model,
                "message": "Query too short after preprocessing"
            }
        
        logger.info(f"Query tokens: {len(query_tokens)} tokens")
        
        with httpx.Client(timeout=30.0) as client:
            retrieval_payload = {
                "query_tokens": query_tokens,
                "query_text": query,
                "model": model,
                "top_k": top_k,
                "include_text": True,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b
            }
            
            if dataset:
                retrieval_payload["dataset"] = dataset
            
            retrieval_resp = client.post(
                f"{RETRIEVAL_SERVICE_URL}/retrieve",
                json=retrieval_payload
            )
            retrieval_resp.raise_for_status()
            retrieval_data = retrieval_resp.json()
            
            logger.info(f"Basic search completed: {retrieval_data.get('total', 0)} results")
            
            return retrieval_data
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during basic search: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error during basic search: {e}", exc_info=True)
        raise


def search_hybrid_serial(
    query: str,
    first_model: str,
    second_model: str,
    first_stage_k: int = 1000,
    final_top_k: int = 50,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    dataset: str = None
) -> dict:
    logger.info(f"Hybrid serial search: {first_model} -> {second_model}, dataset={dataset}")
    logger.info(f"Query: '{query[:50]}...'")
    
    try:
        # Preprocess query with dataset-specific config
        query_tokens = _preprocess_query(query, dataset)
        
        if not query_tokens:
            logger.warning("Preprocessing returned empty tokens")
            return {
                "results": [],
                "total": 0,
                "model": f"hybrid_serial_{first_model}_{second_model}",
                "message": "Query too short after preprocessing"
            }
        
        logger.info(f"Query tokens: {len(query_tokens)} tokens")
        
        with httpx.Client(timeout=30.0) as client:
            retrieval_payload = {
                "query_tokens": query_tokens,
                "query_text": query,
                "first_model": first_model,
                "second_model": second_model,
                "first_stage_k": first_stage_k,
                "final_top_k": final_top_k,
                "include_text": True,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b
            }
            
            if dataset:
                retrieval_payload["dataset"] = dataset
            
            retrieval_resp = client.post(
                f"{RETRIEVAL_SERVICE_URL}/retrieve/hybrid/serial",
                json=retrieval_payload
            )
            retrieval_resp.raise_for_status()
            retrieval_data = retrieval_resp.json()
            
            logger.info(f"Hybrid serial search completed: {retrieval_data.get('total', 0)} results")
            
            return retrieval_data
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during hybrid serial search: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error during hybrid serial search: {e}", exc_info=True)
        raise


def search_hybrid_parallel(
    query: str,
    models: List[str],
    fusion_method: str = "rrf",
    model_weights: Optional[Dict[str, float]] = None,
    top_k: int = 50,
    candidates_per_model: int = 1000,
    rrf_k: int = 60,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    dataset: str = None
) -> dict:
    logger.info(f"Hybrid parallel search: models={models}, fusion={fusion_method}, dataset={dataset}")
    logger.info(f"Query: '{query[:50]}...'")
    
    try:
        # Preprocess query with dataset-specific config
        query_tokens = _preprocess_query(query, dataset)
        
        if not query_tokens:
            logger.warning("Preprocessing returned empty tokens")
            return {
                "results": [],
                "total": 0,
                "model": f"hybrid_parallel_{'_'.join(models)}_{fusion_method}",
                "message": "Query too short after preprocessing"
            }
        
        logger.info(f"Query tokens: {len(query_tokens)} tokens")
        
        with httpx.Client(timeout=60.0) as client:
            retrieval_payload = {
                "query_tokens": query_tokens,
                "query_text": query,
                "models": models,
                "fusion_method": fusion_method,
                "model_weights": model_weights,
                "top_k": top_k,
                "candidates_per_model": candidates_per_model,
                "rrf_k": rrf_k,
                "include_text": True,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b
            }
            
            if dataset:
                retrieval_payload["dataset"] = dataset
            
            retrieval_resp = client.post(
                f"{RETRIEVAL_SERVICE_URL}/retrieve/hybrid/parallel",
                json=retrieval_payload
            )
            retrieval_resp.raise_for_status()
            retrieval_data = retrieval_resp.json()
            
            logger.info(f"Hybrid parallel search completed: {retrieval_data.get('total', 0)} results")
            
            return retrieval_data
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during hybrid parallel search: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error during hybrid parallel search: {e}", exc_info=True)
        raise
